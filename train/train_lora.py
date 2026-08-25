#!/usr/bin/env python3
"""LoRA 파인튜닝 (선택 트랙) — inference 와 물리 분리된 학습 진입점(대회 재현성 규칙).

현재 제출은 **무학습(zero-shot)** 이다. 이 스크립트는 추후 캘리브레이션을 가중치에
직접 학습시키는 트랙의 진입점으로, 폴더·코드를 미리 분리해 둔 것이다.

데이터: build_train_data.py 가 만든 공개 BBQ 기반 SFT JSONL (test/hidden 미사용).
백엔드:
  · 로컬(M4): mlx-vlm LoRA (소형 모델 실험).
  · A6000   : transformers + peft LoRA (bf16, Qwen3-VL-8B). 학습 후 adapter 저장.
학습된 adapter 는 inference 에서 base 모델에 로드해 사용한다(별도 옵션으로 연결 예정).

재현성: seed 고정, 하이퍼파라미터·라이브러리 버전을 adapter 옆 train_manifest.json 에 기록.

예)
  # 로컬 MLX (소형 모델 권장)
  python train/train_lora.py --engine mlx --data train/data/sft.jsonl --iters 500
  # A6000 (transformers+peft)
  python train/train_lora.py --engine peft --data train/data/sft.jsonl --epochs 1
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from skku_vqa import config  # noqa: E402


def train_mlx(args) -> None:
    """mlx-vlm QLoRA(4bit base 동결 + 어댑터 학습). 반드시 래퍼로 실행 —
    ①MRoPE Metal VJP 버그 우회 ②completion-토큰 loss 정규화가 래퍼에 있다.
    (`python -m mlx_vlm lora` 서브커맨드는 0.6.2 에 존재하지 않음 — 주의.)"""
    raise SystemExit(
        "[안내] MLX QLoRA 학습은 래퍼로 실행하세요(권장 하이퍼파라미터 포함):\n"
        f"  HF_HUB_OFFLINE=1 .venv/bin/python train/mlx_lora_wrapper.py \\\n"
        f"    --model-path {args.model or config.DEFAULT_MODEL_MLX} \\\n"
        f"    --dataset train/data/sft_dir --split train --epochs 1 \\\n"
        "    --batch-size 1 --gradient-accumulation-steps 8 \\\n"
        "    --lora-rank 16 --lora-alpha 32 --lora-dropout 0.05 \\\n"
        "    --learning-rate 1e-4 --max-seq-length 1024 --grad-checkpoint \\\n"
        "    --grad-clip 1.0 --train-on-completions --steps-per-save 1000 \\\n"
        f"    --output-path \"$HOME/skku_adapters/run1\"\n"
        "※ 어댑터는 로컬 경로($HOME)에 저장 — Google Drive 동기화 폴더 금지.\n"
        "※ 추론 연결: run_eval/run_inference/run_chunked 의 --adapter-path.")


def train_peft(args) -> None:
    """transformers + peft LoRA (A6000 bf16). 실제 학습 루프 골격."""
    try:
        import torch  # noqa: F401
        from transformers import AutoProcessor, AutoModelForImageTextToText  # noqa: F401
        from peft import LoraConfig  # noqa: F401
    except Exception as e:
        raise SystemExit(f"[scaffold] peft 학습 의존성 미설치: {e}\n"
                         "  pip install -r requirements-a6000.txt (peft 포함) 후 재실행.")
    raise SystemExit(
        "[scaffold] PEFT LoRA 학습 루프는 데이터·하이퍼파라미터 확정 후 구현 예정.\n"
        f"  모델={args.model or config.DEFAULT_MODEL_A6000}  데이터={args.data}\n"
        "  설계: LoraConfig(r=16, alpha=32, target=q/k/v/o_proj) · bf16 · seed 고정\n"
        "        · 어댑터를 out_dir 에 저장 + train_manifest.json 기록.")


def main() -> None:
    ap = argparse.ArgumentParser(description="LoRA 파인튜닝(선택 트랙)")
    ap.add_argument("--engine", choices=["mlx", "peft"], required=True)
    ap.add_argument("--data", required=True, help="build_train_data.py 산출 JSONL")
    ap.add_argument("--model", default=None)
    ap.add_argument("--out", default="train/adapters/lora_v1")
    ap.add_argument("--iters", type=int, default=500)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    Path(args.out).mkdir(parents=True, exist_ok=True)
    manifest = {"engine": args.engine, "data": args.data, "model": args.model,
                "iters": args.iters, "epochs": args.epochs, "seed": args.seed,
                "env": config.env_report()}
    (Path(args.out) / "train_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2))

    if args.engine == "mlx":
        train_mlx(args)
    else:
        train_peft(args)


if __name__ == "__main__":
    main()
