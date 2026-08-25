#!/usr/bin/env python3
"""진단 — 이미지 첨부가 도움인가 독인가? v3b 프롬프트로 동일 케이스를
텍스트 전용 vs 이미지첨부 비교. 규칙상 허용된 '추론 목적' 진단(산출물 미사용).

핵심 질문:
 (A) 진짜 모호(answer=unknown) 케이스에서 이미지를 주면 모델이 외형 보고
     인물을 골라버리나?(=독, 원본 솔루션이 텍스트 전용 택한 이유)
 (B) 텍스트로 이미 풀리는 과기권 케이스에서 이미지가 답을 바꾸나?(중립이어야)
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from skku_vqa import data, parsing, prompts                  # noqa: E402
from skku_vqa.backends import build_backend                  # noqa: E402

OVER_ABSTAIN = ["TEST_5105", "TEST_3123", "TEST_3261", "TEST_7041", "TEST_0075",
                "TEST_3272", "TEST_3725", "TEST_0379", "TEST_5128", "TEST_3208",
                "TEST_0068", "TEST_5818"]
# 진짜 모호(대칭 묘사, 근거 미귀속) — 이미지가 편향 주입하는지 감시(answer=unknown 이 정답)
AMBIG_KEEP = ["TEST_0046", "TEST_0049", "TEST_0056", "TEST_0061"]


def main() -> None:
    pv = "v3b"
    test = data.load_split("test")
    ids = OVER_ABSTAIN + AMBIG_KEEP
    sub = test[test["sample_id"].isin(ids)].set_index("sample_id").loc[ids].reset_index()

    backend = build_backend("mlx", max_new_tokens=256, use_image=True)
    print(f"프롬프트 {pv} · 텍스트 vs 이미지 첨부 (n={len(sub)})\n")
    img_reg = 0       # 모호풀에서 이미지가 인물선택을 유발한 수(=독)
    img_change = 0    # 과기권풀에서 이미지가 답을 바꾼 수
    for _, r in sub.iterrows():
        msgs = prompts.build_messages(r["context"], r["question"],
                                      list(r["options"]), prompt_version=pv)
        raw_t = backend.generate(msgs, image_path=None)
        raw_i = backend.generate(msgs, image_path=r["image_abspath"])
        it, _ = parsing.parse_answer(raw_t, list(r["options"]), r["unknown_index"])
        ii, _ = parsing.parse_answer(raw_i, list(r["options"]), r["unknown_index"])
        u = int(r["unknown_index"]) if pd.notna(r["unknown_index"]) else -1
        pool = "기권풀" if r["sample_id"] in OVER_ABSTAIN else "모호풀"
        tag = ""
        if pool == "모호풀" and ii != u:
            img_reg += 1; tag = "⚠️이미지가 편향주입(인물선택)"
        if pool == "기권풀" and ii != it:
            img_change += 1; tag = "이미지가 답 변경"
        f = lambda x: "기권" if x == u else f"인물{x}"
        print(f"  [{pool}] {r['sample_id']}: 텍스트={f(it)} | 이미지={f(ii)}  {tag}")
    print(f"\n  → 모호풀 {len(AMBIG_KEEP)}건 중 이미지가 편향주입(독): {img_reg}")
    print(f"  → 기권풀 {len(OVER_ABSTAIN)}건 중 이미지가 답 변경: {img_change}")
    print("  해석: 모호풀 편향주입>0 이면 이미지는 위험(텍스트 전용 유지). "
          "기권풀 변경≈0 이면 이미지는 이 슬라이스에 무용.")


if __name__ == "__main__":
    main()
