"""백엔드 공통 인터페이스 + 멀티모달 메시지 포맷 헬퍼.

모든 백엔드는 동일 계약을 따른다:
    generate(messages, image_path=None) -> str
    generate_batch(messages_list, image_paths=None) -> list[str]   # 선택(있으면 배치)

messages 는 prompts.build_messages 가 만든 [{role, content:str}, ...] 형식.
이미지 첨부가 필요하면 각 백엔드가 format_messages 로 typed-parts 로 변환한다.
"""
from __future__ import annotations

from typing import Optional, Protocol, runtime_checkable


@runtime_checkable
class Backend(Protocol):
    """덕 타이핑용 프로토콜. 최소 generate 만 있으면 단건 추론 가능."""

    def generate(self, messages: list[dict], image_path: Optional[str] = None) -> str: ...


def format_messages(messages: list[dict], image_path: Optional[str] = None) -> list[dict]:
    """문자열 content → 멀티모달 프로세서용 typed-parts 리스트로 변환.

    Qwen3-VL 등의 apply_chat_template(tokenize=True) 는 텍스트 전용이라도 content 가
    [{'type':'text', ...}] 형식이어야 내부 이미지 추출 루프에서 TypeError 가 나지
    않는다. 따라서 이미지 유무와 무관하게 항상 typed-parts 로 변환한다.
    image_path 가 주어지면 user 메시지 앞에 이미지 파트를 추가한다.
    """
    out = []
    img = None
    for m in messages:
        parts = ([{"type": "text", "text": m["content"]}]
                 if isinstance(m["content"], str) else list(m["content"]))
        if m["role"] == "user" and image_path:
            from PIL import Image
            if img is None:
                img = Image.open(image_path).convert("RGB")
            parts = [{"type": "image", "image": img}] + parts
        out.append({"role": m["role"], "content": parts})
    return out
