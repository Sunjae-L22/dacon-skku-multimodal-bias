#!/usr/bin/env python3
"""추론 → 제출 CSV 생성 (train 코드와 물리 분리 — 대회 재현성 규칙).

흐름:
  test.csv 로드 → 프롬프트 빌드 → 백엔드 생성 → 텍스트 파싱 → preds(0/1/2)
  → submission.csv 저장 전 무결성 검증 → manifest.json(설정·환경·속도) 동봉.

라벨 규칙: 옵션 순서 = A/B/C 라벨 = 제출 인덱스(재매핑 불필요).
이미지: 1차 제출은 텍스트 전용(--use-image 미지정). 검증셋과 동일 조건.

예)
  # GPU 없이 배관 점검(Mock, 상위 50행)
  python inference/run_inference.py --backend mock:always_unknown --limit 50
  # A6000 최종 제출(vLLM, 전체 8500, v2, 텍스트 전용)
  python inference/run_inference.py --backend vllm:Qwen/Qwen3-VL-8B-Instruct --batch
  # 로컬 M4 단건(MLX)
  python inference/run_inference.py --backend mlx --limit 24
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

# pipeline 루트를 import 경로에 추가(설치 없이 동작).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from skku_vqa import config, data, runner, submission, prompts  # noqa: E402
from skku_vqa.backends import build_backend                      # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="test.csv → submission.csv")
    ap.add_argument("--backend", default="mock:always_unknown",
                    help="mock:* | mlx[:id] | vllm:id | transformers:id")
    ap.add_argument("--prompt", dest="prompt_version", default="v2",
                    choices=list(prompts.PROMPTS.keys()))
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--limit", type=int, default=None, help="상위 N행만(배관 점검)")
    # --batch 기본은 미지정(None) → vLLM 백엔드면 자동 켠다(연속배치=0.5s/샘플 규칙의 핵심).
    # --no-batch 로 강제 단건(디버그)도 가능. vLLM 을 단건으로 돌리면 규칙을 어기므로 자동화한다.
    ap.add_argument("--batch", dest="batch", action="store_true", default=None,
                    help="generate_batch 경로(vLLM 권장). vLLM 이면 미지정 시 자동 ON")
    ap.add_argument("--no-batch", dest="batch", action="store_false",
                    help="강제 단건(디버그용, 비권장)")
    ap.add_argument("--use-image", action="store_true", help="이미지 첨부(1차 비권장)")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--adapter-path", default=None,
                    help="LoRA 어댑터 디렉토리(mlx 전용, train/ 트랙 산출물)")
    ap.add_argument("--load-in-4bit", action="store_true",
                    help="transformers 백엔드 4bit(bitsandbytes nf4) — 작은 VRAM(3060 Ti 8GB)")
    ap.add_argument("--batch-size", type=int, default=8,
                    help="transformers 미니배치 크기(8GB면 4~8). vLLM 은 무관")
    # --- vLLM 전용(클라우드 A10G/L4 24GB, Kaggle 2×T4 등) ---
    ap.add_argument("--quantization", default=None,
                    help="vLLM 양자화: 30B-A3B-AWQ 는 'awq_marlin'(Ampere). bf16 모델은 생략")
    ap.add_argument("--dtype", default=None, help="vLLM/transformers dtype(기본 auto)")
    ap.add_argument("--max-model-len", type=int, default=None, help="vLLM 컨텍스트 상한")
    ap.add_argument("--gpu-mem-util", type=float, default=None,
                    help="vLLM gpu_memory_utilization(24GB 빠듯하면 0.95)")
    ap.add_argument("--tensor-parallel-size", type=int, default=None,
                    help="vLLM 멀티GPU(Kaggle 2×T4 면 2)")
    ap.add_argument("--out", default=None, help="출력 경로(기본 outputs/submission.csv)")
    args = ap.parse_args()

    kind, _, marg = args.backend.partition(":")
    if args.batch is None:                       # 미지정: GPU 배치형 백엔드는 자동 배치
        args.batch = kind in ("vllm", "transformers", "hf")
    # manifest 에 기록할 실제 모델 가중치 해소(재현성).
    if kind in ("vllm", "transformers", "hf"):
        resolved_model = marg or config.DEFAULT_MODEL_A6000
    elif kind == "mlx":
        resolved_model = marg or config.DEFAULT_MODEL_MLX
    else:
        resolved_model = None

    cfg = config.RunConfig(
        backend=args.backend, model_id=resolved_model,
        prompt_version=args.prompt_version, use_image=args.use_image,
        max_new_tokens=args.max_new, seed=args.seed, batch_mode=bool(args.batch),
        adapter_path=args.adapter_path)

    # vLLM 전용 kwargs 는 지정된 것만 전달(나머지 백엔드는 **extra 로 흡수).
    extra_kw = {}
    if args.quantization is not None: extra_kw["quantization"] = args.quantization
    if args.dtype is not None: extra_kw["dtype"] = args.dtype
    if args.max_model_len is not None: extra_kw["max_model_len"] = args.max_model_len
    if args.gpu_mem_util is not None: extra_kw["gpu_memory_utilization"] = args.gpu_mem_util
    if args.tensor_parallel_size is not None:
        extra_kw["tensor_parallel_size"] = args.tensor_parallel_size
    backend = build_backend(args.backend, max_new_tokens=args.max_new,
                            use_image=args.use_image, seed=args.seed,
                            adapter_path=args.adapter_path,
                            load_in_4bit=args.load_in_4bit, batch_size=args.batch_size,
                            **extra_kw)

    df = data.load_split("test")
    if args.limit is not None:
        df = df.head(args.limit).copy()

    t0 = time.perf_counter()
    res = runner.predict_dataframe(
        df, backend, prompt_version=args.prompt_version, use_image=args.use_image,
        batch=args.batch, desc=f"TEST({args.prompt_version})")
    elapsed = time.perf_counter() - t0

    preds = res["preds"]
    sub = pd.DataFrame({"sample_id": df["sample_id"].tolist(),
                        "label": [int(p) for p in preds]})

    # 진단: '모름' 옵션 선택 비율(검증셋 ambiguous 비중과 sanity 비교용).
    # ※ unknown_index 컬럼에 None 이 섞이면 pandas 가 float64(NaN)로 올리므로 pd.notna 로 가드.
    unk_hits = sum(1 for p, u in zip(preds, df["unknown_index"].tolist())
                   if pd.notna(u) and int(p) == int(u))
    unknown_pred_rate = unk_hits / len(preds) if preds else 0.0

    stats = submission.validate_submission(sub, check_ref=(args.limit is None))

    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out) if args.out else (config.OUTPUT_DIR / "submission.csv")
    sub.to_csv(out_path, index=False)

    per_sample = elapsed / len(df) if len(df) else 0.0
    manifest = {
        "run_config": cfg.metadata(),
        "env": config.env_report(),
        "n_samples": len(df),
        "elapsed_sec": round(elapsed, 2),
        "sec_per_sample": round(per_sample, 4),
        "meets_0.5s_rule": per_sample <= 0.5,
        "parse_ok_rate": res["parse_ok_rate"],
        "n_unparsed": res["n_unparsed"],
        "unknown_pred_rate": unknown_pred_rate,
        "label_dist": stats["label_dist"],
        "matches_sample_submission": stats["matches_sample_submission"],
    }
    manifest_path = out_path.with_name(out_path.stem + "_manifest.json")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))

    print(f"\n[제출 생성] {out_path}")
    print(f"  행수            : {stats['n_rows']}")
    print(f"  label 분포      : {stats['label_dist']}")
    print(f"  parse_ok        : {res['parse_ok_rate']*100:.1f}%  (미파싱 {res['n_unparsed']})")
    print(f"  '모름' 예측 비율: {unknown_pred_rate*100:.1f}%")
    print(f"  속도            : {per_sample*1000:.0f} ms/샘플  "
          f"({'0.5s 규칙 충족' if per_sample <= 0.5 else '⚠️ 0.5s 초과'})")
    print(f"  sample_submission 일치: {stats['matches_sample_submission']}")
    print(f"  manifest        : {manifest_path}")

    # 규칙 가드(소프트 경고 — 비채점 하드웨어에선 자연히 초과하므로 차단하지 않는다).
    if kind == "vllm" and not args.batch:
        print("  ⚠️  vLLM 을 단건(--no-batch)으로 실행 — 0.5s/샘플 규칙 위반 위험. --batch 권장.")
    if args.limit is None and not manifest["meets_0.5s_rule"]:
        print("  ⚠️  평균 속도가 0.5s/샘플을 초과 — 기준 환경(A6000)에서 재측정 필요.")
    if res["n_unparsed"]:
        print(f"  ⚠️  LLM 텍스트 미파싱 {res['n_unparsed']}건은 unknown 으로 폴백됨"
              "(최종답=LLM 텍스트 원칙상 0 이 이상적 — 프롬프트/토큰 점검).")


if __name__ == "__main__":
    main()
