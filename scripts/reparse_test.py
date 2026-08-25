#!/usr/bin/env python3
"""저장된 test 청크의 raw(모델 생성 텍스트)를 현행 파서로 재파싱해 제출본을 재생성.

모델 재실행 없이 파서 개선분만 반영한다 — 규칙 준수: 답은 여전히 LLM 이 생성한
텍스트에서 추출되며, 재파싱은 추출 단계의 결함 수정일 뿐이다.

사용: python3 scripts/reparse_test.py [--prompt v2]
  → outputs/submission_reparsed.csv (+ 기존 submission.csv 와의 diff 출력)
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from skku_vqa import config, data, parsing, submission  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="v2", help="재파싱할 청크의 프롬프트 버전")
    args = ap.parse_args()

    name = "test" if args.prompt == "v2" else f"test_{args.prompt}"
    paths = sorted(glob.glob(str(config.OUTPUT_DIR / "chunks" / name / "*.csv")))
    if not paths:
        raise SystemExit(f"청크 없음: outputs/chunks/{name}/")
    chunks = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    chunks = chunks.sort_values("row_index").reset_index(drop=True)

    test = data.load_split("test")
    assert len(chunks) == len(test), f"청크 {len(chunks)} != test {len(test)}"
    opts = dict(zip(test["sample_id"], test["options"]))
    unks = dict(zip(test["sample_id"], test["unknown_index"]))

    new_preds, n_flip, n_unparsed = [], 0, 0
    for _, r in chunks.iterrows():
        sid = r["sample_id"]
        idx, ok = parsing.parse_answer(str(r["raw"]), opts[sid], unks[sid])
        new_preds.append(idx)
        n_unparsed += (not ok)
        if idx != int(r["pred"]):
            n_flip += 1
            print(f"  FLIP {sid}: {int(r['pred'])} → {idx}")

    sub = pd.DataFrame({"sample_id": chunks["sample_id"].tolist(),
                        "label": [int(p) for p in new_preds]})
    stats = submission.validate_submission(sub, check_ref=True)
    out = config.OUTPUT_DIR / "submission_reparsed.csv"
    sub.to_csv(out, index=False)
    print(f"\n[재파싱 제출본] {out}")
    print(f"  플립 {n_flip}건 · 미파싱 {n_unparsed}건 · 행수 {stats['n_rows']} · "
          f"sample_submission 일치 {stats['matches_sample_submission']}")


if __name__ == "__main__":
    main()
