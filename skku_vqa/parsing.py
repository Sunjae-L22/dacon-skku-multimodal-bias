"""LLM 출력 텍스트 → 옵션 인덱스 추출.

규칙 준수: 최종 답은 LLM 이 생성한 텍스트에서 '추출'한다(기계적 포맷팅).
`Final answer: <LETTER>` 를 최우선으로 보고, 단계적 폴백을 둔다. 모든 단계가
실패할 때만 unknown 옵션으로 폴백하고 그 사실(parsed_ok=False)을 보고해
폴백 비율을 로깅할 수 있게 한다.
"""
from __future__ import annotations

import re
from typing import Optional

LETTER2IDX = {"A": 0, "B": 1, "C": 2}

# 'Final answer: B' / 'final answer - (B)' / 'Answer: C' 등 관용 표기.
# 셋째 패턴: 반추 루프로 Final answer 줄 없이 잘린 출력의 본문 결론
# ("Therefore, the correct answer is B") — test 미파싱 9건 중 2건이 이 형태로
# 실점 확인(TEST_1026/7894). Final answer 류가 전부 없을 때만 발동.
_FINAL_PATTERNS = [
    r"final\s*answer\s*[:\-–]?\s*\(?\s*([ABC])\b",
    r"\banswer\s*[:\-–]?\s*\(?\s*([ABC])\b",
    r"\banswer\s+is\s*[:\"'*]*\s*\(?\s*([ABC])\b",
]
_PAREN_LETTER = re.compile(r"\(?\b([ABC])\)")


def parse_answer(text: str, options: list[str],
                 unknown_index: Optional[int]) -> tuple[int, bool]:
    """모델 출력에서 선택지 인덱스를 뽑는다. 반환: (index, parsed_ok).

    parsed_ok=False 면 LLM 텍스트에서 답을 못 읽어 unknown 폴백을 쓴 것.
    """
    if not text or not str(text).strip():
        return (unknown_index if unknown_index is not None else 0), False
    t = str(text).strip()

    # 1) 'Final answer: X' / 'Answer: X' — 마지막 매치 우선(모델이 여러 번 언급 시 결론).
    for pat in _FINAL_PATTERNS:
        m = list(re.finditer(pat, t, flags=re.IGNORECASE))
        if m:
            return LETTER2IDX[m[-1].group(1).upper()], True

    # 2) 끝부분의 단독 'X)' 또는 '(X)' (마지막 매치).
    m = list(_PAREN_LETTER.finditer(t))
    if m:
        return LETTER2IDX[m[-1].group(1).upper()], True

    # 3) 옵션 본문 문자열이 출력에 그대로 등장하면 매칭(가장 긴 옵션 우선 — 부분포함 오인 방지).
    low = t.lower()
    order = sorted(range(len(options)), key=lambda i: -len(str(options[i])))
    for i in order:
        opt = str(options[i]).strip().lower()
        if opt and opt in low:
            return i, True

    # 4) 폴백: unknown 옵션(의미식별), 없으면 0.
    return (unknown_index if unknown_index is not None else 0), False
