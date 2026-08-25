#!/usr/bin/env python3
"""재개 가능한 청크 추론 러너 — 장시간 로컬 MLX 실행의 메모리 thrash·중단에 대비.

문제: M4 단건 MLX 루프는 샘플이 누적되면(>~100) Metal 버퍼 캐시가 쌓여 속도가
폭주(thrash)하고, 풀 dev(1988, ~3.3h)·test(8500, ~12h)는 도중에 죽을 수 있다.

해법: 데이터를 청크로 쪼개 **청크마다 별도 프로세스**로 추론한다. 프로세스가
끝나면 OS 가 메모리를 회수하므로 thrash 가 누적되지 않는다. 각 청크 결과를 CSV 로
저장하고, 이미 끝난 청크는 건너뛴다(**resume**). 모든 청크가 끝나면 집계한다:
  · eval(smoke/dev/full) → Balanced Accuracy + 오류분석
  · test                 → submission.csv + 무결성검증 + manifest

사용:
  # 풀 dev 측정(레퍼런스 ~0.97 대조), 80개씩, 죽으면 같은 명령으로 이어서
  .venv/bin/python scripts/run_chunked.py --backend mlx --target dev --chunk 80
  # 1차 제출(test 8500) — 밤샘. 중단돼도 재실행하면 남은 청크만 처리
  .venv/bin/python scripts/run_chunked.py --backend mlx --target test --chunk 100
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from skku_vqa import config, data, runner, metrics, submission, prompts  # noqa: E402
from skku_vqa.backends import build_backend                              # noqa: E402

EVAL_TARGETS = {"smoke", "dev", "full", "hard", "val"}


def load_target_df(target: str) -> pd.DataFrame:
    if target in EVAL_TARGETS:
        return data.load_validation(target)
    if target == "test":
        return data.load_split("test")
    raise SystemExit(f"unknown target: {target} (smoke/dev/full/hard/val/test)")


def chunk_path(target: str, prompt_version: str, start: int, end: int,
               adapter_path: str | None = None) -> Path:
    # 프롬프트 버전·어댑터별 디렉토리 분리(A/B 시 청크 재사용 충돌 방지).
    # v2 무어댑터는 기존 레이아웃 유지 → 저장된 dev/test v2 청크에서 그대로 resume.
    name = target if prompt_version == "v2" else f"{target}_{prompt_version}"
    if adapter_path:
        name += f"_ad-{Path(adapter_path).name}"
    d = config.OUTPUT_DIR / "chunks" / name
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{start:06d}-{end:06d}.csv"


# --------------------------------------------------------------------------- #
# worker: 한 청크[start:end] 만 추론해 CSV 로 저장(원자적 쓰기)
# --------------------------------------------------------------------------- #
def run_worker(args) -> None:
    df = load_target_df(args.target).iloc[args.start:args.end].copy()
    backend = build_backend(args.backend, max_new_tokens=args.max_new,
                            use_image=args.use_image, seed=args.seed,
                            adapter_path=args.adapter_path)
    res = runner.predict_dataframe(
        df, backend, prompt_version=args.prompt_version, use_image=args.use_image,
        batch=False, desc=f"{args.target}[{args.start}:{args.end}]")

    out = pd.DataFrame({
        "row_index": range(args.start, args.start + len(df)),
        "sample_id": df["sample_id"].tolist(),
        "pred": [int(p) for p in res["preds"]],
        "parse_ok": [bool(o) for o in res["oks"]],
        "raw": res["raw"],
    })
    for col in ("label", "is_ambiguous", "category", "unknown_index"):
        if col in df.columns:
            out[col] = df[col].tolist()

    dest = Path(args.out)
    tmp = dest.with_suffix(".tmp.csv")
    out.to_csv(tmp, index=False)
    tmp.replace(dest)                     # 원자적 — 중단돼도 반쪽 파일이 안 남음


# --------------------------------------------------------------------------- #
# orchestrator: 청크 분할 → 미완료 청크만 서브프로세스로 실행 → 집계
# --------------------------------------------------------------------------- #
def _chunk_done(path: Path, expected: int) -> bool:
    if not path.exists():
        return False
    try:
        return len(pd.read_csv(path)) == expected
    except Exception:
        return False


def run_orchestrator(args) -> None:
    df = load_target_df(args.target)
    n = len(df)
    ranges = [(s, min(s + args.chunk, n)) for s in range(0, n, args.chunk)]
    print(f"[chunked] target={args.target} n={n} chunk={args.chunk} "
          f"→ {len(ranges)} chunks · backend={args.backend}")

    t0 = time.perf_counter()
    done = 0
    for i, (s, e) in enumerate(ranges):
        cp = chunk_path(args.target, args.prompt_version, s, e, args.adapter_path)
        if _chunk_done(cp, e - s):
            done += 1
            continue
        cmd = [sys.executable, str(Path(__file__).resolve()), "--worker",
               "--target", args.target, "--start", str(s), "--end", str(e),
               "--backend", args.backend, "--prompt", args.prompt_version,
               "--max-new", str(args.max_new), "--seed", str(args.seed),
               "--out", str(cp)]
        if args.use_image:
            cmd.append("--use-image")
        if args.adapter_path:
            cmd += ["--adapter-path", str(args.adapter_path)]
        print(f"  chunk {i+1}/{len(ranges)} [{s}:{e}] → 실행", flush=True)
        r = subprocess.run(cmd)
        if r.returncode != 0 or not _chunk_done(cp, e - s):
            print(f"\n[중단] chunk [{s}:{e}] 실패(rc={r.returncode}). "
                  "같은 명령으로 재실행하면 이 청크부터 이어집니다.")
            raise SystemExit(1)
        done += 1
        el = time.perf_counter() - t0
        eta = el / done * (len(ranges) - done) if done else 0
        print(f"    ✓ {done}/{len(ranges)} chunks · 경과 {el/60:.1f}분 · ETA {eta/60:.1f}분",
              flush=True)

    # ---- 집계 ----
    parts = [pd.read_csv(chunk_path(args.target, args.prompt_version, s, e,
                                    args.adapter_path))
             for s, e in ranges]
    full = pd.concat(parts, ignore_index=True).sort_values("row_index").reset_index(drop=True)
    assert len(full) == n, f"집계 행수 불일치: {len(full)} != {n}"
    n_unparsed = int((~full["parse_ok"].astype(bool)).sum())

    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.target in EVAL_TARGETS:
        _aggregate_eval(args, full, n_unparsed)
    else:
        _aggregate_test(args, df, full, n_unparsed, time.perf_counter() - t0)


def _aggregate_eval(args, full: pd.DataFrame, n_unparsed: int) -> None:
    sys.path.insert(0, str(ROOT / "eval"))
    import analyze_errors  # noqa: E402
    m = metrics.balanced_accuracy(
        full["label"].astype(int).tolist(), full["pred"].astype(int).tolist(),
        full["is_ambiguous"].tolist())
    print(f"\n[검증 결과] target={args.target} prompt={args.prompt_version} "
          f"backend={args.backend}  (n={m['n']})")
    print(f"  Balanced Accuracy : {m['balanced_accuracy']:.4f}")
    print(f"    · ambiguous     : {m['acc_ambiguous']:.4f}  (n={m['n_ambiguous']})")
    print(f"    · disambiguated : {m['acc_disambiguated']:.4f}  (n={m['n_disambiguated']})")
    print(f"  parse_ok          : {(1 - n_unparsed/len(full))*100:.1f}%  (미파싱 {n_unparsed})")
    analyze_errors.print_report(full)

    pred_path = config.OUTPUT_DIR / f"eval_{args.target}_{args.prompt_version}_pred.csv"
    full.to_csv(pred_path, index=False)
    err = analyze_errors.classify_errors(full)
    err = err[err["error_type"] != "CORRECT"]
    err_path = config.OUTPUT_DIR / f"eval_{args.target}_{args.prompt_version}_errors.csv"
    err.to_csv(err_path, index=False)
    print(f"\n  저장: {pred_path.name} · {err_path.name}")


def _aggregate_test(args, df, full, n_unparsed, elapsed) -> None:
    sub = pd.DataFrame({"sample_id": full["sample_id"].tolist(),
                        "label": full["pred"].astype(int).tolist()})
    stats = submission.validate_submission(sub, check_ref=True)
    out_path = config.OUTPUT_DIR / "submission.csv"
    sub.to_csv(out_path, index=False)

    kind, _, marg = args.backend.partition(":")
    model_id = (marg or config.DEFAULT_MODEL_MLX) if kind == "mlx" else \
               (marg or config.DEFAULT_MODEL_A6000) if kind in ("vllm", "transformers", "hf") else None
    manifest = {
        "run_config": config.RunConfig(
            backend=args.backend, model_id=model_id,
            prompt_version=args.prompt_version, use_image=args.use_image,
            max_new_tokens=args.max_new, seed=args.seed, batch_mode=False,
            adapter_path=args.adapter_path).metadata(),
        "env": config.env_report(),
        "n_samples": len(df), "elapsed_sec": round(elapsed, 1),
        "sec_per_sample": round(elapsed / len(df), 4),
        "parse_ok_rate": (1 - n_unparsed / len(full)),
        "n_unparsed": n_unparsed, "label_dist": stats["label_dist"],
        "matches_sample_submission": stats["matches_sample_submission"],
        "chunked": True, "chunk_size": args.chunk,
    }
    (out_path.with_name("submission_manifest.json")).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f"\n[제출 생성] {out_path}")
    print(f"  행수 {stats['n_rows']} · label 분포 {stats['label_dist']}")
    print(f"  parse_ok {(1-n_unparsed/len(full))*100:.1f}% (미파싱 {n_unparsed})")
    print(f"  sample_submission 일치 {stats['matches_sample_submission']}")
    if n_unparsed:
        print(f"  ⚠️  미파싱 {n_unparsed}건 unknown 폴백 — 프롬프트/토큰 점검 권장")


def main() -> None:
    ap = argparse.ArgumentParser(description="재개 가능한 청크 추론(eval/제출)")
    ap.add_argument("--worker", action="store_true", help="(내부용) 단일 청크 처리")
    ap.add_argument("--backend", default="mlx")
    ap.add_argument("--target", default="dev", help="smoke|dev|full|test")
    ap.add_argument("--chunk", type=int, default=80)
    ap.add_argument("--prompt", dest="prompt_version", default="v2",
                    choices=list(prompts.PROMPTS.keys()))
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--use-image", action="store_true")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--adapter-path", default=None,
                    help="LoRA 어댑터 디렉토리(mlx 전용, train/ 트랙 산출물)")
    # worker 전용
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.worker:
        run_worker(args)
    else:
        run_orchestrator(args)


if __name__ == "__main__":
    main()
