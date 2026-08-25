#!/usr/bin/env python3
"""검증 하니스 — 공개 BBQ 검증셋으로 Balanced Accuracy 를 오프라인 측정한다.

평가셋은 ambiguous 여부가 비공개지만, 자체 BBQ 검증셋은 라벨이 있어 대회 산식을
그대로 잰다. smoke(204, 빠른 점검) / dev(1988, 본측정) / full 분리.

예)
  python eval/run_eval.py --backend mock:always_unknown --set smoke
  python eval/run_eval.py --backend mlx --set dev --save
  python eval/run_eval.py --backend vllm:Qwen/Qwen3-VL-8B-Instruct --set dev --batch --save
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from skku_vqa import config, data, runner, metrics, prompts  # noqa: E402
from skku_vqa.backends import build_backend                   # noqa: E402

# eval/ 안의 analyze_errors 모듈을 직접 import.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_errors  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="BBQ 검증셋 Balanced Accuracy 측정")
    ap.add_argument("--backend", default="mock:always_unknown")
    ap.add_argument("--set", dest="which", default="dev",
                    choices=["smoke", "dev", "full", "hard", "val"])
    ap.add_argument("--prompt", dest="prompt_version", default="v2",
                    choices=list(prompts.PROMPTS.keys()))
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch", action="store_true")
    ap.add_argument("--use-image", action="store_true")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--adapter-path", default=None,
                    help="LoRA 어댑터 디렉토리(mlx 전용, train/ 트랙 산출물)")
    # vLLM 전용(클라우드 24GB 등) — run_inference.py 와 동일.
    ap.add_argument("--quantization", default=None,
                    help="vLLM 양자화(30B-A3B-AWQ 는 'awq_marlin')")
    ap.add_argument("--dtype", default=None)
    ap.add_argument("--max-model-len", type=int, default=None)
    ap.add_argument("--gpu-mem-util", type=float, default=None)
    ap.add_argument("--tensor-parallel-size", type=int, default=None)
    ap.add_argument("--save", action="store_true",
                    help="예측·오류 CSV 를 outputs/ 에 저장")
    args = ap.parse_args()

    extra_kw = {}
    if args.quantization is not None: extra_kw["quantization"] = args.quantization
    if args.dtype is not None: extra_kw["dtype"] = args.dtype
    if args.max_model_len is not None: extra_kw["max_model_len"] = args.max_model_len
    if args.gpu_mem_util is not None: extra_kw["gpu_memory_utilization"] = args.gpu_mem_util
    if args.tensor_parallel_size is not None:
        extra_kw["tensor_parallel_size"] = args.tensor_parallel_size
    backend = build_backend(args.backend, max_new_tokens=args.max_new,
                            use_image=args.use_image, seed=args.seed,
                            adapter_path=args.adapter_path, **extra_kw)

    df = data.load_validation(args.which)
    if args.limit is not None:
        df = df.head(args.limit).copy()

    t0 = time.perf_counter()
    res = runner.predict_dataframe(
        df, backend, prompt_version=args.prompt_version, use_image=args.use_image,
        batch=args.batch, desc=f"EVAL:{args.which}({args.prompt_version})")
    elapsed = time.perf_counter() - t0

    df = df.copy()
    df["pred"] = res["preds"]
    df["raw"] = res["raw"]

    m = metrics.balanced_accuracy(
        df["label"].astype(int).tolist(), df["pred"].astype(int).tolist(),
        df["is_ambiguous"].tolist())

    print(f"\n[검증 결과] set={args.which} prompt={args.prompt_version} "
          f"backend={args.backend}  (n={m['n']})")
    print(f"  Balanced Accuracy : {m['balanced_accuracy']:.4f}")
    print(f"    · ambiguous     : {m['acc_ambiguous']:.4f}  (n={m['n_ambiguous']})")
    print(f"    · disambiguated : {m['acc_disambiguated']:.4f}  (n={m['n_disambiguated']})")
    print(f"  parse_ok          : {res['parse_ok_rate']*100:.1f}%  (미파싱 {res['n_unparsed']})")
    print(f"  속도              : {elapsed/len(df)*1000:.0f} ms/샘플")

    analyze_errors.print_report(df)

    if args.save:
        config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        pred_path = config.OUTPUT_DIR / f"eval_{args.which}_{args.prompt_version}_pred.csv"
        cols = ["sample_id", "category", "is_ambiguous", "label", "pred",
                "unknown_index", "raw"]
        df[[c for c in cols if c in df.columns]].to_csv(pred_path, index=False)
        err = analyze_errors.classify_errors(df)
        err = err[err["error_type"] != "CORRECT"]
        err_path = config.OUTPUT_DIR / f"eval_{args.which}_{args.prompt_version}_errors.csv"
        err[[c for c in cols + ["error_type"] if c in err.columns]].to_csv(
            err_path, index=False)
        print(f"\n  저장: {pred_path.name} · {err_path.name}")


if __name__ == "__main__":
    main()
