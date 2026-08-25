#!/usr/bin/env python3
"""두 평가 결과(pred CSV)를 sample_id 로 맞춰 flip(고침/회귀)을 집계한다.

hard 셋 A/B 의 판독기: 절대 점수가 아니라 '기준(v2) 대비 무엇이 바뀌었나'를 본다.
  · fixed      — 기준이 틀리고 후보가 맞힘 (오류 교정)
  · regressed  — 기준이 맞히고 후보가 틀림 (회귀 — 가장 경계)
  · both_wrong — 둘 다 틀림 (답이 서로 다를 수 있음)

사용:
  python3 scripts/compare_runs.py outputs/eval_dev_v2_pred.csv outputs/eval_hard_v3a_pred.csv
(첫 인자 = 기준 baseline, 둘째 인자 = 후보. 교집합 sample_id 만 비교)
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


def main(base_path: str, cand_path: str) -> None:
    base = pd.read_csv(base_path)[["sample_id", "pred", "label", "is_ambiguous", "category"]]
    cand = pd.read_csv(cand_path)[["sample_id", "pred"]]
    m = base.merge(cand, on="sample_id", suffixes=("_base", "_cand"))
    if m.empty:
        raise SystemExit("교집합 sample_id 가 없습니다 — 인자 순서/파일 확인")

    m["ok_base"] = m["pred_base"].astype(int) == m["label"].astype(int)
    m["ok_cand"] = m["pred_cand"].astype(int) == m["label"].astype(int)
    m["flip"] = "same"
    m.loc[~m["ok_base"] & m["ok_cand"], "flip"] = "fixed"
    m.loc[m["ok_base"] & ~m["ok_cand"], "flip"] = "regressed"
    m.loc[~m["ok_base"] & ~m["ok_cand"], "flip"] = "both_wrong"

    n_fix = int((m["flip"] == "fixed").sum())
    n_reg = int((m["flip"] == "regressed").sum())
    print(f"[비교] base={Path(base_path).name}  cand={Path(cand_path).name}  (교집합 {len(m)})")
    print(f"  고침(fixed)     : {n_fix}")
    print(f"  회귀(regressed) : {n_reg}   ← 0 에 가까워야 함")
    print(f"  순이득(net)     : {n_fix - n_reg:+d}")

    for grp, name in [(True, "ambiguous"), (False, "disambiguated")]:
        g = m[m["is_ambiguous"].astype(bool) == grp]
        print(f"\n  [{name}] n={len(g)}  base 정답 {int(g['ok_base'].sum())} → cand 정답 {int(g['ok_cand'].sum())}")
        sub = g[g["flip"].isin(["fixed", "regressed"])]
        if not sub.empty:
            print(pd.crosstab(sub["category"], sub["flip"]).to_string())

    reg = m[m["flip"] == "regressed"]
    if not reg.empty:
        print("\n  회귀 케이스 sample_id:", ", ".join(map(str, reg["sample_id"].tolist())))


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    main(sys.argv[1], sys.argv[2])
