"""GPU 불필요 모의 백엔드 — 배관(프롬프트→생성→파싱→채점→제출) 검증·테스트용.

strategy:
  · always_unknown : unknown 옵션을 고른다(전부 '모름' 베이스라인 / ambiguous 상한 체크).
  · first          : 항상 A.
  · random         : 시드 고정 무작위(결정적).
  · oracle         : 정답이 messages 에 없으니, 옵션 텍스트만으론 못 맞힘 → unknown 과 동일.

최종 답 형식은 실제 모델과 같게 'Final answer: X' 로 내보내 parser 를 실제와 동일 경로로 검증.
"""
from __future__ import annotations

import random
import re
from typing import Optional

from .. import config

_OPT_RE = re.compile(r"^([ABC])\)\s*(.+)$", re.MULTILINE)


def _user_text(messages: list[dict]) -> str:
    user = messages[-1]["content"]
    if isinstance(user, str):
        return user
    return " ".join(c.get("text", "") for c in user if isinstance(c, dict))


def _options_from_messages(messages: list[dict]) -> list[str]:
    text = _user_text(messages)
    found = {m.group(1): m.group(2).strip() for m in _OPT_RE.finditer(text)}
    return [found.get(L, "") for L in ("A", "B", "C")]


class MockBackend:
    def __init__(self, strategy: str = "always_unknown", seed: int = 42):
        self.strategy = strategy
        self.rng = random.Random(seed)

    def _pick_letter(self, options: list[str]) -> str:
        if self.strategy == "first":
            return "A"
        if self.strategy == "random":
            return "ABC"[self.rng.randint(0, 2)]
        # always_unknown / oracle: 의미식별로 unknown 옵션을 고른다.
        for i, o in enumerate(options):
            if any(k in str(o).lower() for k in config.UNKNOWN_LEXICON):
                return "ABC"[i]
        return "C"

    def generate(self, messages: list[dict], image_path: Optional[str] = None) -> str:
        letter = self._pick_letter(_options_from_messages(messages))
        return f"Based on the context (mock reasoning).\nFinal answer: {letter}"

    def generate_batch(self, messages_list: list[list[dict]],
                       image_paths=None) -> list[str]:
        return [self.generate(m) for m in messages_list]
