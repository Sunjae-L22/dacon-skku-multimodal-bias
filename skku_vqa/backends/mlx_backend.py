"""Apple Silicon(M-series) 로컬 추론용 MLX 백엔드 — 빠른 반복(개발) 전용.

· 모델: mlx-community/Qwen3-VL-8B-Instruct-4bit (A6000 제출과 같은 Qwen3-VL-8B 의
  4bit MLX 변환). 여기서 얻은 프롬프트 캘리브레이션 결과가 A6000 bf16 제출로 잘 전이된다.
· 인터페이스 동일: generate(messages, image_path=None) -> str.
· 기본 greedy(temperature=0.0)로 재현성. 텍스트 주도(use_image=False).
· 단건 ~5s/샘플(M4). 최종 제출 속도(0.5s)는 vLLM 으로 별도 충족 — MLX 는 개발용.

설치: pip install -r requirements-mac.txt   (mlx, mlx-vlm GitHub 최신)
※ MLX 는 Apple Silicon 전용 — 리눅스/CUDA 에선 import 불가(정상).
"""
from __future__ import annotations

from typing import Optional


class MLXVLMBackend:
    def __init__(self, model_id: str = "mlx-community/Qwen3-VL-8B-Instruct-4bit",
                 max_new_tokens: int = 256, temperature: float = 0.0,
                 use_image: bool = False, clear_cache_every: int = 16,
                 adapter_path: Optional[str] = None, **extra):
        from mlx_vlm import load, generate as _gen   # Apple Silicon 에서만 import 성공
        self._gen = _gen
        # adapter_path: train/ 트랙이 학습한 LoRA 어댑터 디렉토리(adapters.safetensors
        # + adapter_config.json). 지정 시 base 위에 비융합 로드(mlx_vlm 정식 지원).
        if adapter_path:
            self.model, self.processor = load(model_id, adapter_path=str(adapter_path))
        else:
            self.model, self.processor = load(model_id)
        self.adapter_path = adapter_path
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.use_image = use_image
        # ★장시간 단건 루프 시 Metal 버퍼 캐시가 누적돼 속도가 폭주(thrash)하는 것을 막기 위해
        #  주기적으로 mx 캐시를 비운다(결과 불변 — 메모리만 회수). 버전별 함수명 방어.
        self._gen_count = 0
        self._clear_every = max(0, int(clear_cache_every))
        self._mx_clear = None
        try:
            import mlx.core as mx
            self._mx_clear = getattr(mx, "clear_cache", None) or \
                getattr(getattr(mx, "metal", None), "clear_cache", None)
        except Exception:
            self._mx_clear = None

    def _tick(self) -> None:
        """generate 호출 카운트 → 주기적으로 Metal 캐시 비움(메모리 thrash 방지)."""
        self._gen_count += 1
        if self._mx_clear and self._clear_every and \
                self._gen_count % self._clear_every == 0:
            try:
                self._mx_clear()
            except Exception:
                pass
        # 이미지 경로용 config(이미지 토큰 삽입에 필요).
        try:
            from mlx_vlm.utils import load_config
            self._config = load_config(model_id)
        except Exception:
            self._config = getattr(self.model, "config", None)

    def _prompt(self, messages: list[dict]) -> str:
        """HF 프로세서 chat template 로 시스템+유저 → 프롬프트 문자열(텍스트 전용).
        시스템 역할을 보존하려고 mlx-vlm 의 apply_chat_template 대신 프로세서를 직접 쓴다."""
        return self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False)

    def generate(self, messages: list[dict], image_path: Optional[str] = None) -> str:
        self._tick()                              # 주기적 캐시 회수(thrash 방지)
        use_img = image_path if (self.use_image and image_path) else None
        if use_img is not None:
            from mlx_vlm.prompt_utils import apply_chat_template
            prompt = apply_chat_template(self.processor, self._config, messages,
                                         num_images=1)
            out = self._gen(self.model, self.processor, prompt, image=[use_img],
                            max_tokens=self.max_new_tokens, verbose=False)
            return str(getattr(out, "text", out)).strip()

        prompt = self._prompt(messages)           # 텍스트 전용(검증된 경로)
        common = dict(max_tokens=self.max_new_tokens, verbose=False)
        # mlx-vlm 버전별 인자명 차이(temperature/temp) 방어. greedy 기본.
        for kw in ("temperature", "temp"):
            try:
                out = self._gen(self.model, self.processor, prompt,
                                **{kw: self.temperature}, **common)
                break
            except TypeError:
                continue
        else:
            out = self._gen(self.model, self.processor, prompt, **common)
        return str(getattr(out, "text", out)).strip()
