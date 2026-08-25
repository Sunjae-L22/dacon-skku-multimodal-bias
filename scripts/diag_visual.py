#!/usr/bin/env python3
"""진단 전용 — 시각-템플릿 과기권 케이스에서 프롬프트 v0/v3b 가 v2 대비
기권→인물 로 교정되는지(그리고 진짜 모호 케이스는 기권 유지하는지) 확인한다.

규칙 준수: test 에 대한 추론(=허용된 '추론 목적' 사용)일 뿐, 어떤 산출물도
학습/프롬프트 재료로 쓰지 않는다(룰 8 Data Leakage 무관). 결과는 화면 출력만.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from skku_vqa import config, data, parsing, prompts          # noqa: E402
from skku_vqa.backends import build_backend                  # noqa: E402

# 분석에서 고른 진단 표본(하드코딩 — 재현용).
OVER_ABSTAIN = [  # 텍스트로 풀리는데 v2 가 기권한 disamb 시각템플릿
    "TEST_5105", "TEST_3123", "TEST_3261", "TEST_7041", "TEST_0075", "TEST_3272",
    "TEST_3725", "TEST_0379", "TEST_5128", "TEST_3208", "TEST_0068", "TEST_5818",
]
AMBIG_KEEP = [  # 진짜 모호 — 기권이 정답이어야 함(회귀 감시)
    "TEST_0046", "TEST_0049", "TEST_0056", "TEST_0061", "TEST_0057", "TEST_0010",
]


def main() -> None:
    import os
    import time
    prompt_versions = [a for a in sys.argv[1:] if not a.startswith("-")] or ["v0", "v3b"]
    # 모델 교체: SKKU_MLX_MODEL 환경변수(예: mlx-community/Qwen3-VL-30B-A3B-Instruct-4bit)
    model = os.environ.get("SKKU_MLX_MODEL")
    spec = f"mlx:{model}" if model else "mlx"
    test = data.load_split("test")
    ids = OVER_ABSTAIN + AMBIG_KEEP
    sub = test[test["sample_id"].isin(ids)].set_index("sample_id").loc[ids].reset_index()

    for pv in prompt_versions:
        t_load = time.perf_counter()
        backend = build_backend(spec, max_new_tokens=256)
        print(f"\n{'='*70}\n프롬프트 {pv} · 모델 {model or 'Qwen3-VL-8B-4bit(기본)'} "
              f"· 로드 {time.perf_counter()-t_load:.0f}s\n{'='*70}")
        t_gen = time.perf_counter()
        n_fix, n_reg = 0, 0
        for _, r in sub.iterrows():
            msgs = prompts.build_messages(r["context"], r["question"],
                                          list(r["options"]), prompt_version=pv)
            raw = backend.generate(msgs)
            idx, ok = parsing.parse_answer(raw, list(r["options"]), r["unknown_index"])
            is_unk = (pd.notna(r["unknown_index"]) and idx == int(r["unknown_index"]))
            group = "기권풀" if r["sample_id"] in OVER_ABSTAIN else "모호풀"
            if r["sample_id"] in OVER_ABSTAIN and not is_unk:
                n_fix += 1
            if r["sample_id"] in AMBIG_KEEP and not is_unk:
                n_reg += 1
            mark = "★고침" if (r["sample_id"] in OVER_ABSTAIN and not is_unk) else \
                   ("⚠️회귀" if (r["sample_id"] in AMBIG_KEEP and not is_unk) else "")
            print(f"  [{group}] {r['sample_id']}: idx{idx} "
                  f"{'(기권)' if is_unk else '(인물선택)'} {mark}")
        dt = time.perf_counter() - t_gen
        print(f"\n  → 기권풀 12건 중 교정(인물선택): {n_fix}/12")
        print(f"  → 모호풀 6건 중 회귀(인물선택): {n_reg}/6  (0 이 이상적)")
        print(f"  → 속도 {dt/len(sub):.1f}s/샘플 (n={len(sub)})")


if __name__ == "__main__":
    main()
