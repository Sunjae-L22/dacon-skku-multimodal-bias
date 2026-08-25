"""mlx_vlm.lora 실행 래퍼 — 두 가지 필수 보정을 적용한 학습 진입점.

① MRoPE Metal 커널 우회: mlx-vlm 0.6.2 의 융합 Metal MRoPE 커널
   (models/rope_utils.py)에 VJP 가 없어 학습이 'Primitive::vjp Not implemented
   for CustomKernel' 로 즉사한다. 모델 생성 전에 _HAS_METAL=False 를 설정하면
   순수 MLX(미분 가능) 경로로 폴백한다. 추론은 별도 프로세스라 영향 없음.

② completion-토큰 정규화 loss: 기본 vision_language_loss_fn 은 completion 만
   학습(--train-on-completions)할 때도 손실을 **전체 실토큰 수**로 나눈다.
   우리 타깃('Final answer: X' ≈ 6 tok)은 프롬프트(~230 tok)에 희석돼 gradient
   가 ~1/40 로 줄어 '어댑터가 안 먹는' 실패가 거의 확정 — completion 토큰 수로
   나누도록 교체한다(--no-completion-norm 으로 끄고 LR 보상 폴백 가능).

사용(예 — 본 학습):
  cd pipeline && HF_HUB_OFFLINE=1 ./.venv/bin/python train/mlx_lora_wrapper.py \
    --model-path mlx-community/Qwen3-VL-8B-Instruct-4bit \
    --dataset train/data/sft_dir --split train --epochs 1 \
    --batch-size 1 --gradient-accumulation-steps 8 \
    --lora-rank 16 --lora-alpha 32 --lora-dropout 0.05 \
    --learning-rate 1e-4 --max-seq-length 1024 --grad-checkpoint --grad-clip 1.0 \
    --train-on-completions --steps-per-report 25 --steps-per-save 1000 \
    --output-path "$HOME/skku_adapters/run1"
※ 어댑터 출력은 반드시 로컬 경로($HOME 등) — Google Drive 동기화 폴더 금지.
"""

import functools

import mlx.core as mx
import mlx.nn as nn
import numpy as np

import mlx_vlm.models.rope_utils as rope_utils

rope_utils._HAS_METAL = False          # ① 모델 생성 전에 설정해야 함

from mlx_vlm import lora               # noqa: E402  (_HAS_METAL 설정 이후 import)
from mlx_vlm.trainer import sft_trainer  # noqa: E402


def completion_normalized_loss_fn(model, batch, train_on_completions=False,
                                  assistant_id=77091):
    """sft_trainer.vision_language_loss_fn 동일 로직 + 정규화만 교정.

    원본: ce.sum() / length_mask.sum()  ← 분모가 '전체 실토큰'이라 짧은 타깃 희석.
    교정: ce.sum() / (weight_mask * length_mask).sum()  ← completion 토큰 수.
    """
    pixel_values = batch["pixel_values"]
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    batch_size, seq_length = input_ids.shape

    if train_on_completions:
        weight_mask = mx.ones_like(attention_mask)
        assistant_response_index = np.full((batch_size,), -1, dtype=np.int32)
        input_ids_np = np.array(input_ids)
        for row_idx, row in enumerate(input_ids_np):
            positions = np.where(row == assistant_id)[0]
            if positions.size > 0:
                assistant_response_index[row_idx] = positions[0]
        range_matrix = mx.repeat(
            mx.expand_dims(mx.arange(seq_length), 0), batch_size, axis=0)
        assistant_mask = range_matrix <= mx.array(assistant_response_index).reshape(-1, 1)
        weight_mask = mx.where(assistant_mask, mx.zeros_like(weight_mask),
                               weight_mask)[:, 1:]
    else:
        weight_mask = None

    input_ids = input_ids[:, :-1]
    attention_mask = attention_mask[:, :-1]
    lengths = mx.sum(attention_mask, axis=1)
    labels = batch["input_ids"][:, 1:]

    kwargs = {k: v for k, v in batch.items()
              if k not in ["input_ids", "pixel_values", "attention_mask"]}
    outputs = model(input_ids, pixel_values, attention_mask, **kwargs)
    logits = outputs.logits.astype(mx.float32)

    if logits.shape[1] < labels.shape[1]:
        pad = ((0, 0), (0, labels.shape[1] - logits.shape[1]), (0, 0))
        logits = mx.pad(logits, pad, mode="constant", constant_values=-100)
    elif logits.shape[1] > labels.shape[1]:
        logits = logits[:, -labels.shape[1]:, :]

    seq_len = input_ids.shape[1]
    lengths = mx.minimum(lengths, seq_len)
    length_mask = mx.arange(seq_len)[None, :] < lengths[:, None]

    ce = nn.losses.cross_entropy(logits, labels, weights=weight_mask) * length_mask
    if weight_mask is not None:                      # ★ 교정 지점
        denom = mx.maximum((weight_mask * length_mask).sum(), 1)
    else:
        denom = mx.maximum(length_mask.sum(), 1)
    return ce.sum() / denom


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train Vision-Language Model")
    parser.add_argument("--model-path", type=str, default="mlx-community/Qwen2-VL-2B-Instruct-bf16")
    parser.add_argument("--full-finetune", action="store_true")
    parser.add_argument("--train-vision", action="store_true")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--dataset-config", type=str, default=None)
    parser.add_argument("--image-resize-shape", type=int, nargs=2, default=None)
    parser.add_argument("--custom-prompt-format", type=str, default=None)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--iters", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--steps-per-report", type=int, default=10)
    parser.add_argument("--steps-per-eval", type=int, default=200)
    parser.add_argument("--steps-per-save", type=int, default=100)
    parser.add_argument("--val-batches", type=int, default=4)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--grad-checkpoint", action="store_true")
    parser.add_argument("--grad-clip", type=float, default=None)
    parser.add_argument("--train-on-completions", action="store_true")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--assistant-id", type=int, default=77091)
    parser.add_argument("--lora-alpha", type=float, default=16)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--train-mode", type=str, default="sft", choices=["sft", "orpo"])
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--output-path", type=str, default="adapters.safetensors")
    parser.add_argument("--adapter-path", type=str, default=None)
    parser.add_argument("--no-completion-norm", action="store_true",
                        help="② loss 정규화 패치 끔(폴백: LR 을 ~5e-4 로 보상)")

    args = parser.parse_args()

    if not args.no_completion_norm:
        # lora.main 은 모듈 전역 'train' 을 호출 시점에 조회하므로 rebind 가 적용된다.
        lora.train = functools.partial(sft_trainer.train,
                                       loss_fn=completion_normalized_loss_fn)
        print("[wrapper] completion-토큰 정규화 loss 적용 (--no-completion-norm 으로 해제)")
    print("[wrapper] MRoPE Metal 커널 비활성(_HAS_METAL=False) — 학습 미분 가능 경로")

    lora.main(args)
