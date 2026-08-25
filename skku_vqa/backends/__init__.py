"""백엔드 팩토리. spec 문자열 → 백엔드 인스턴스.

spec 형식: "<kind>[:<arg>]"
  · mock:always_unknown | mock:first | mock:random
  · mlx                 | mlx:<hf_id>            (로컬 M4, 개발용)
  · vllm:<hf_id>        | vllm                   (A6000 최종 제출, 0.5s/샘플)
  · transformers:<hf_id> | hf:<hf_id>            (A6000 단건 폴백/대조용)

kwargs(max_new_tokens, temperature, use_image, seed 등)는 그대로 백엔드에 전달.
import 는 지연(lazy)시켜 mlx/vllm 미설치 환경에서도 다른 백엔드를 쓸 수 있게 한다.
"""
from __future__ import annotations

from .. import config


def build_backend(spec: str, **kwargs):
    kind, _, arg = spec.partition(":")
    # LoRA 어댑터(MLX 전용). vLLM 의 Qwen3-VL+LoRA 는 업스트림 버그로 미지원 —
    # A6000 검증 경로는 병합(HF bf16) 모델을 쓴다(train/merge_export 참고).
    adapter_path = kwargs.pop("adapter_path", None)

    if kind == "mock":
        from .mock import MockBackend
        return MockBackend(strategy=arg or "always_unknown")

    if kind == "mlx":
        from .mlx_backend import MLXVLMBackend
        return MLXVLMBackend(arg or config.DEFAULT_MODEL_MLX,
                             adapter_path=adapter_path, **kwargs)

    if adapter_path:
        raise NotImplementedError(
            f"adapter_path 는 mlx 백엔드 전용입니다(요청: {kind}). "
            "A6000 추론은 병합된 HF 모델을 사용하세요.")

    if kind == "vllm":
        from .vllm_backend import VLLMBackend
        return VLLMBackend(arg or config.DEFAULT_MODEL_A6000, **kwargs)

    if kind in ("transformers", "hf"):
        from .transformers_backend import TransformersVLMBackend
        return TransformersVLMBackend(arg or config.DEFAULT_MODEL_A6000, **kwargs)

    raise NotImplementedError(
        f"백엔드 '{kind}' 미구현. 지원: mock / mlx / vllm / transformers")
