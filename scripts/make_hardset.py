#!/usr/bin/env python3
"""hard 검증셋 생성 — 프롬프트 A/B 를 풀 dev(47분) 대신 ~12분에 측정하기 위한 표본.

구성(결정적, seed 고정):
  · dev 오류 전수            60건  (DISAMBIG_ABSTAIN 44 / AMBIG_OVERCOMMIT 11 / DISAMBIG_WRONG 5)
  · ambiguous 정답(회귀감시)  60건  — Age 15건 우선(과진입 회귀 최대 위험) + 카테고리 비례
  · disambiguated 정답(〃)    30건  — 카테고리 비례
  · 위험 풀(risk_pool)       ~40건 — v3 적대 검증이 지목한 회귀 채널 표본:
      비귀속 차이("one of them…"), sat-down attractive, Disability 정체성 기술,
      직업/계급만 차이 + 능력/재력 질문(SES·Race_x_SES), Age 좌석·경험차 쌍
→ validation/bbq_hard.csv (hard_role 컬럼으로 출처 표시)

해석 기준: 후보 프롬프트가 '오류 60' 을 몇 건 고치고 '정답 90' 을 몇 건 깨는지.
hard 셋은 오류를 의도적으로 과표집했으므로 여기서의 balanced accuracy 절대값은
의미가 없다 — flip 수(고침/회귀)만 본다. 최종 판단은 풀 dev 로 확정.

사용: python3 scripts/make_hardset.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from skku_vqa import config  # noqa: E402

SEED = 1234
N_AMB_OK, N_AMB_OK_AGE = 60, 15
N_DIS_OK = 30


def _stratified_sample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """카테고리 비례 표본(결정적). 반올림 잔여분은 큰 카테고리부터 채움."""
    if len(df) <= n:
        return df
    picks = []
    sizes = df["category"].value_counts()
    quota = (sizes / sizes.sum() * n).round().astype(int).clip(lower=1)
    for cat, k in quota.items():
        g = df[df["category"] == cat]
        picks.append(g.sample(min(k, len(g)), random_state=seed))
    out = pd.concat(picks)
    if len(out) > n:
        out = out.sample(n, random_state=seed)
    elif len(out) < n:
        rest = df.drop(out.index)
        out = pd.concat([out, rest.sample(n - len(out), random_state=seed)])
    return out


def main() -> None:
    dev = pd.read_csv(config.BBQ_DEV)
    pred = pd.read_csv(config.OUTPUT_DIR / "eval_dev_v2_pred.csv")
    err_ids = set(pd.read_csv(config.OUTPUT_DIR / "eval_dev_v2_errors.csv")["sample_id"])

    ok = pred[pred["pred"].astype(int) == pred["label"].astype(int)]
    amb_ok = dev[dev["sample_id"].isin(ok["sample_id"]) & dev["is_ambiguous"].astype(bool)
                 & ~dev["sample_id"].isin(err_ids)]
    dis_ok = dev[dev["sample_id"].isin(ok["sample_id"]) & ~dev["is_ambiguous"].astype(bool)
                 & ~dev["sample_id"].isin(err_ids)]

    # ambiguous 회귀감시: Age 를 먼저 고정 표집(과진입 회귀 최대 위험) + 나머지 비례.
    amb_age = amb_ok[amb_ok["category"] == "Age"].sample(
        min(N_AMB_OK_AGE, (amb_ok["category"] == "Age").sum()), random_state=SEED)
    amb_rest = _stratified_sample(amb_ok.drop(amb_age.index), N_AMB_OK - len(amb_age), SEED)

    # v3 적대 검증이 지목한 ambiguous 회귀 채널 — 현재 정답인 표본을 명시 포함.
    q = dev["question"].astype(str)
    ctx = dev["context"].astype(str)
    pools = [
        amb_ok[ctx.loc[amb_ok.index].str.contains(
            r"[Oo]ne of the(m| two)|[Oo]nly one|very different", regex=True)].head(12),
        amb_ok[q.loc[amb_ok.index].str.contains("attractive", case=False)].head(10),
        amb_ok[amb_ok["category"] == "Disability_status"].head(8),
        amb_ok[amb_ok["category"].isin(["SES", "Race_x_SES"]) & q.loc[amb_ok.index].str.contains(
            r"intelligen|smart|capable|afford|parent|[Ee]nglish|speak|educat|money|wealth",
            regex=True)].head(12),
        dev[dev["sample_id"].isin(["BBQ-Age-001036", "BBQ-Age-001038",
                                   "BBQ-Sexual_orientation-058356",
                                   "BBQ-Sexual_orientation-058358"])],
    ]
    risk = pd.concat(pools).drop_duplicates(subset="sample_id")

    parts = {
        "error": dev[dev["sample_id"].isin(err_ids)],
        "amb_ok": pd.concat([amb_age, amb_rest]),
        "dis_ok": _stratified_sample(dis_ok, N_DIS_OK, SEED),
    }
    used = pd.concat(parts.values())["sample_id"]
    parts["risk_pool"] = risk[~risk["sample_id"].isin(set(used))]
    hard = pd.concat([p.assign(hard_role=role) for role, p in parts.items()],
                     ignore_index=True)
    assert hard["sample_id"].is_unique, "hard 셋 sample_id 중복"

    config.BBQ_HARD.parent.mkdir(parents=True, exist_ok=True)
    hard.to_csv(config.BBQ_HARD, index=False)
    print(f"[hard 셋 생성] {config.BBQ_HARD}  (n={len(hard)})")
    print(hard.groupby("hard_role").size().to_string())
    print("\n카테고리 분포:")
    print(pd.crosstab(hard["category"], hard["hard_role"]).to_string())


if __name__ == "__main__":
    main()
