"""🤗 Transformers 기반 VLM 백엔드 — CUDA 추론(배칭 지원).

용도: ① 작은 VRAM CUDA GPU(예: 집 RTX 3060 Ti 8GB)에서 4bit 배치 추론 →
M4 MLX 단건 대비 10~20배. ② A6000 채점환경과 같은 CUDA/Ampere 경로라 **재현성**
(룰⑦) 확보. vLLM 미가용(Windows 네이티브 등) 환경의 대량 추론 1급 경로.

· generate_batch: 내부에서 batch_size 단위로 미니배치(좌패딩) 처리 → 처리량↑.
· 기본 greedy(do_sample=False) → 결정적·재현 가능. 최종 답은 LLM 생성 텍스트.
· 4bit(bitsandbytes nf4): 8GB GPU 에 8B 적재. Ampere(sm86)에서 A6000 과 동일 커널.

규칙: 공개일 2026-06-01 이전 + Apache-2.0(Qwen3-VL) → 충족. git-source 의존 없음.
"""
from __future__ import annotations

from typing import Optional

from .base import format_messages


class TransformersVLMBackend:
    def __init__(self, model_id: str, load_in_4bit: bool = False, dtype: str = "auto",
                 device_map: str = "auto", max_new_tokens: int = 256,
                 do_sample: bool = False, temperature: float = 0.7, top_p: float = 0.8,
                 top_k: int = 20, use_image: bool = False, batch_size: int = 8,
                 attn_implementation: Optional[str] = None, **extra):
        import torch
        from transformers import AutoProcessor, AutoModelForImageTextToText
        self.torch = torch
        self.model_id = model_id
        self.use_image = use_image
        self.max_new_tokens = max_new_tokens
        self.do_sample = do_sample
        self.batch_size = max(1, int(batch_size))
        self.gen_kwargs = (dict(temperature=temperature, top_p=top_p, top_k=top_k)
                           if do_sample else {})

        model_kwargs = dict(device_map=device_map)
        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            compute_dtype = (torch.bfloat16 if torch.cuda.is_bf16_supported()
                             else torch.float16)
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=compute_dtype, bnb_4bit_use_double_quant=True)
        else:
            model_kwargs["dtype"] = dtype  # transformers>=4.57: torch_dtype → dtype
        if attn_implementation:
            model_kwargs["attn_implementation"] = attn_implementation

        self.processor = AutoProcessor.from_pretrained(model_id)
        # 배치 generate 는 decoder-only 라 좌패딩 필수(우패딩이면 출력이 깨짐).
        tok = getattr(self.processor, "tokenizer", None)
        if tok is not None:
            tok.padding_side = "left"
            if tok.pad_token_id is None and tok.eos_token_id is not None:
                tok.pad_token = tok.eos_token
        self.pad_id = (tok.pad_token_id if tok is not None else None)
        self.model = AutoModelForImageTextToText.from_pretrained(model_id, **model_kwargs)
        self.model.eval()

    def _gen_one_batch(self, convs: list[list[dict]]) -> list[str]:
        inputs = self.processor.apply_chat_template(
            convs, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt", padding=True).to(self.model.device)
        with self.torch.no_grad():
            gen = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens,
                                      do_sample=self.do_sample, pad_token_id=self.pad_id,
                                      **self.gen_kwargs)
        trimmed = gen[:, inputs["input_ids"].shape[1]:]
        return [self.processor.decode(t, skip_special_tokens=True).strip()
                for t in trimmed]

    def generate_batch(self, messages_list: list[list[dict]],
                       image_paths: Optional[list] = None) -> list[str]:
        """전체를 batch_size 단위 미니배치로 처리(좌패딩). 입력 순서 보존."""
        try:
            from tqdm.auto import tqdm
        except Exception:
            def tqdm(it=None, **k): return it
        n = len(messages_list)
        outs: list[str] = []
        for s in tqdm(range(0, n, self.batch_size), desc="gen", unit="batch"):
            chunk = messages_list[s:s + self.batch_size]
            convs = []
            for j, msgs in enumerate(chunk):
                ip = (image_paths[s + j] if (self.use_image and image_paths) else None)
                convs.append(format_messages(msgs, ip))
            outs.extend(self._gen_one_batch(convs))
        return outs

    def generate(self, messages: list[dict], image_path: Optional[str] = None) -> str:
        ip = image_path if self.use_image else None
        return self._gen_one_batch([format_messages(messages, ip)])[0]
