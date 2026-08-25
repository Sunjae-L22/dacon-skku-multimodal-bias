"""vLLM 백엔드 — 최종 제출(A6000 48GB) 용. 0.5초/샘플 규칙 충족의 핵심.

왜 vLLM 인가:
  · Transformers 단건 generate 는 A6000 에서도 8B 기준 1~2초/샘플이라 규칙 초과.
  · vLLM 은 PagedAttention + 연속 배치(continuous batching) 로 8,500개를 한 번에
    스케줄링한다. 짧은 context(중앙값 ~27단어) + 짧은 출력(<=256토큰, greedy)이라
    8B bf16 이 A6000 에서 평균 0.5초/샘플을 넉넉히 만족한다(실측은 inference 에서 로깅).

규칙 준수:
  · greedy(temperature=0) → 결정적·재현 가능.
  · 최종 답은 여전히 LLM 생성 텍스트(parser 가 추출). 백엔드는 텍스트만 만든다.

설계: generate_batch 가 1급 경로(전체를 llm.chat 한 번에 넘김). generate(단건)는 래퍼.
이미지(use_image=True)는 OpenAI 스타일 content 파트로 첨부하나, 1차는 텍스트 전용 기본.

※ vLLM 은 CUDA 전용 — Apple Silicon/CPU 에선 import 불가(정상). 로컬은 MLX 사용.
"""
from __future__ import annotations

import base64
from typing import Optional


def _to_image_url_messages(messages: list[dict], image_path: str) -> list[dict]:
    """user 메시지 텍스트 앞에 base64 data-URL 이미지 파트를 끼운다(vLLM chat 규약)."""
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    data_url = f"data:image/jpeg;base64,{b64}"
    out = []
    for m in messages:
        if m["role"] == "user" and isinstance(m["content"], str):
            out.append({"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": data_url}},
                {"type": "text", "text": m["content"]},
            ]})
        else:
            out.append(m)
    return out


class VLLMBackend:
    """vLLM 기반 VLM 추론. 기본 Qwen/Qwen3-VL-8B-Instruct (bf16, Apache-2.0)."""

    def __init__(self,
                 model_id: str = "Qwen/Qwen3-VL-8B-Instruct",
                 max_new_tokens: int = 256,
                 temperature: float = 0.0,
                 seed: int = 1234,
                 use_image: bool = False,
                 dtype: str = "auto",
                 quantization: Optional[str] = None,
                 gpu_memory_utilization: float = 0.90,
                 max_model_len: int = 4096,
                 max_num_seqs: int = 256,
                 tensor_parallel_size: int = 1,
                 **extra):
        from vllm import LLM, SamplingParams  # CUDA 전용
        self._SamplingParams = SamplingParams
        self.model_id = model_id
        self.use_image = use_image
        self.max_new_tokens = max_new_tokens

        # dtype: bf16 모델은 'auto'(=bfloat16). AWQ/GPTQ 4bit 모델은 'auto'가
        # 알아서 float16 compute 선택. quantization='awq_marlin'(Ampere) 등을 명시하면
        # vLLM 이 4bit 커널을 강제(30B-A3B-AWQ 24GB 적재용). 미지정 시 모델 config 자동감지.
        llm_kwargs = dict(
            model=model_id,
            dtype=dtype,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            max_num_seqs=max_num_seqs,
            tensor_parallel_size=tensor_parallel_size,
            seed=seed,
            trust_remote_code=True,
        )
        if quantization:
            llm_kwargs["quantization"] = quantization
        if use_image:
            # 프롬프트당 이미지 1장 상한(메모리/스케줄러 힌트).
            llm_kwargs["limit_mm_per_prompt"] = {"image": 1}
        # 다른 백엔드용 공통 kwargs 는 vLLM LLM() 이 모르므로 제거(에러 방지).
        for k in ("load_in_4bit", "batch_size", "adapter_path"):
            extra.pop(k, None)
        llm_kwargs.update(extra)
        # VRAM 빠듯한 큰 모델(32B-4bit ~19.6GB on 24GB)용: CUDA 그래프 비활성으로
        # ~1GB 회수 → KV 캐시 확보. env 로 제어(코드 재배포 없이). 약간 느려지나 적재 가능.
        import os
        if os.environ.get("VLLM_ENFORCE_EAGER") == "1":
            llm_kwargs["enforce_eager"] = True
        self.llm = LLM(**llm_kwargs)

        # greedy: temperature=0 → vLLM 이 argmax. 재현성.
        self.sampling = SamplingParams(
            temperature=temperature,
            top_p=1.0,
            max_tokens=max_new_tokens,
            seed=seed,
        )

    def generate_batch(self, messages_list: list[list[dict]],
                       image_paths: Optional[list] = None) -> list[str]:
        """전체 대화 리스트를 한 번에 llm.chat → 연속배치로 처리. 입력 순서 보존."""
        if self.use_image and image_paths is not None:
            convs = [_to_image_url_messages(m, p) if p else m
                     for m, p in zip(messages_list, image_paths)]
        else:
            convs = list(messages_list)
        outputs = self.llm.chat(convs, sampling_params=self.sampling, use_tqdm=True)
        # vLLM 은 입력 순서를 보존해 RequestOutput 리스트를 돌려준다.
        return [o.outputs[0].text.strip() for o in outputs]

    def generate(self, messages: list[dict], image_path: Optional[str] = None) -> str:
        return self.generate_batch([messages],
                                   [image_path] if image_path else None)[0]
