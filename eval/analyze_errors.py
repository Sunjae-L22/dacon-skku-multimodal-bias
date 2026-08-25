#!/usr/bin/env python3
"""오류 분석 — 오답을 유형별·카테고리별로 분해해 다음 개선 레버를 도출한다.

오류 유형(이 대회의 핵심 진단):
  · DISAMBIG_ABSTAIN  : 근거(disambiguated)가 있는데 '모름'을 골라 실점 ← 주 병목.
  · DISAMBIG_WRONG    : 근거가 있는데 엉뚱한 인물을 고름(고정관념/오독).
  · AMBIG_OVERCOMMIT  : 근거가 없는데(ambiguous) 특정 인물을 골라 실점(과신).
  · CORRECT           : 정답.

입력 DataFrame 필요 컬럼: label, pred, is_ambiguous, unknown_index, (category 선택).
이 모듈은 run_eval 에서 호출되거나, 저장된 예측 CSV 에 직접 적용할 수 있다.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


def classify_row(label: int, pred: int, is_ambiguous: bool, unknown_index) -> str:
    if int(pred) == int(label):
        return "CORRECT"
    if is_ambiguous:
        # ambiguous 정답은 '모름'. 특정 인물을 고른 것 = 과신.
        return "AMBIG_OVERCOMMIT"
    # disambiguated 오답. ※ unknown_index 가 NaN(None→float 승격)일 수 있어 pd.notna 가드.
    if pd.notna(unknown_index) and int(pred) == int(unknown_index):
        return "DISAMBIG_ABSTAIN"
    return "DISAMBIG_WRONG"


def classify_errors(df: pd.DataFrame) -> pd.DataFrame:
    """error_type 컬럼을 붙여 반환(원본 비변경)."""
    df = df.copy()
    df["error_type"] = [
        classify_row(r["label"], r["pred"], bool(r["is_ambiguous"]), r.get("unknown_index"))
        for _, r in df.iterrows()
    ]
    return df


def summarize(df: pd.DataFrame) -> dict:
    """유형 분포 + 카테고리별 정확도(약점 카테고리 식별)."""
    d = classify_errors(df)
    out: dict = {"type_counts": d["error_type"].value_counts().to_dict()}
    d["correct"] = (d["pred"].astype(int) == d["label"].astype(int))
    if "category" in d.columns:
        cat = d.groupby("category")["correct"].mean().sort_values()
        out["category_accuracy"] = {k: round(float(v), 4) for k, v in cat.items()}
        out["weakest_categories"] = list(cat.head(5).index)
    out["n_errors"] = int((~d["correct"]).sum())
    out["n"] = len(d)
    return out


def print_report(df: pd.DataFrame) -> None:
    s = summarize(df)
    print(f"\n[오류 분석] 전체 {s['n']} · 오답 {s['n_errors']}")
    print("  유형 분포:")
    for k, v in sorted(s["type_counts"].items(), key=lambda kv: -kv[1]):
        print(f"    {k:18s}: {v}")
    if "category_accuracy" in s:
        print("  약한 카테고리(정확도 낮은 순 top5):")
        for c in s["weakest_categories"]:
            print(f"    {c:24s}: {s['category_accuracy'][c]:.3f}")


if __name__ == "__main__":
    # 저장된 예측 CSV(label,pred,is_ambiguous,unknown_index,category) 직접 분석.
    if len(sys.argv) < 2:
        print("usage: python eval/analyze_errors.py <pred_csv>")
        raise SystemExit(1)
    df = pd.read_csv(sys.argv[1])
    print_report(df)
