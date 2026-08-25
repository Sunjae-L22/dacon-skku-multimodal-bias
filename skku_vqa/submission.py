"""제출 파일 무결성 검증. 문제가 있으면 AssertionError 로 즉시 중단(잘못된 제출 방지).

검사: 컬럼=[sample_id,label] · 결측 0 · label∈{0..N-1} 정수 · sample_id 중복 0
      · (전체 실행 시) sample_submission 과 행수·순서·값 일치.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from . import config


def validate_submission(sub: pd.DataFrame,
                        n_options: int = config.N_OPTIONS,
                        sample_path: Optional[Path] = None,
                        check_ref: bool = True) -> dict:
    assert list(sub.columns) == ["sample_id", "label"], \
        f"컬럼이 [sample_id,label] 이어야 함: {list(sub.columns)}"
    assert sub["sample_id"].notna().all(), "sample_id 결측 존재"
    assert sub["label"].notna().all(), "label 결측 존재"

    # 정수성 선검사: 1.5 같은 비정수가 astype(int) 절단으로 통과하는 것을 막는다.
    assert all(float(v) == int(v) for v in sub["label"]), \
        f"label 은 정수여야 함(절단 금지): {[v for v in sub['label'] if float(v) != int(v)][:5]}"
    labels = sub["label"].astype(int)
    assert labels.between(0, n_options - 1).all(), \
        f"label 범위 위반(허용 0..{n_options-1}): {sorted(set(labels))}"
    assert sub["sample_id"].is_unique, "sample_id 중복 존재"

    stats = {
        "n_rows": len(sub),
        "label_dist": labels.value_counts().sort_index().to_dict(),
    }

    sp = sample_path or config.SAMPLE_SUBMISSION
    if check_ref and Path(sp).exists():
        ref = pd.read_csv(sp)
        assert len(sub) == len(ref), f"행수 불일치: sub={len(sub)} ref={len(ref)}"
        assert sub["sample_id"].tolist() == ref["sample_id"].tolist(), \
            "sample_id 순서/값이 sample_submission 과 불일치"
        stats["matches_sample_submission"] = True
    else:
        stats["matches_sample_submission"] = None
    return stats
