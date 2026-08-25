"""데이터 로더 — CSV 파싱 + 옵션 리스트 + unknown 옵션 의미식별 + 이미지 경로 헬퍼.

대회 데이터(test/train)와 자체 BBQ 검증셋을 같은 인터페이스(DataFrame + 파생
컬럼)로 올린다. ``answers`` → 옵션 리스트, unknown 인덱스는 의미(부분문자열)로
식별한다(규칙기반 위치/표현 탐지가 막혀 있으므로).
"""
from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Optional

import pandas as pd

from . import config


def parse_options(raw) -> list[str]:
    """answers 컬럼(JSON 문자열 또는 이미 리스트)을 옵션 리스트로 파싱."""
    if isinstance(raw, list):
        return raw
    try:
        return json.loads(raw)
    except Exception:
        return ast.literal_eval(raw)  # 작은따옴표 등 비표준 JSON 폴백


def find_unknown_index(options: list[str],
                       lexicon: Optional[list[str]] = None) -> Optional[int]:
    """옵션 중 '모름/판단불가'에 해당하는 인덱스를 의미(부분문자열)로 찾는다.

    매칭이 정확히 1개면 그 인덱스, 0개거나 2개 이상이면 None(=모호) 반환.
    ※ 이 값은 정답이 아니라 파싱 폴백·진단용으로만 쓴다(규칙 준수).
    """
    lex = lexicon if lexicon is not None else config.UNKNOWN_LEXICON
    hits = [i for i, o in enumerate(options)
            if any(k in str(o).lower() for k in lex)]
    return hits[0] if len(hits) == 1 else None


def _attach_derived(df: pd.DataFrame, img_dir: Optional[Path]) -> pd.DataFrame:
    """공통 파생 컬럼(options, unknown_index, image_abspath)을 붙인다."""
    df = df.copy()
    df["options"] = df["answers"].apply(parse_options)
    df["unknown_index"] = df["options"].apply(find_unknown_index)
    if img_dir is not None and "image_path" in df.columns:
        df["image_abspath"] = df["image_path"].apply(
            lambda p: str(img_dir / Path(str(p)).name)
        )
    else:
        df["image_abspath"] = None
    return df


def load_split(split: str) -> pd.DataFrame:
    """대회 데이터 'train' 또는 'test' 로드 + 파생 컬럼."""
    assert split in ("train", "test"), f"unknown split: {split}"
    if split == "train":
        csv_path, img_dir = config.TRAIN_CSV, config.TRAIN_IMG_DIR
    else:
        csv_path, img_dir = config.TEST_CSV, config.TEST_IMG_DIR
    if not Path(csv_path).exists():
        raise FileNotFoundError(
            f"데이터 CSV 없음: {csv_path}\n"
            f"  DATA_DIR={config.DATA_DIR} (환경변수 SKKU_DATA_DIR 로 덮어쓸 수 있음)\n"
            f"  기대 구조: <DATA_DIR>/{split}/{split}.csv , <DATA_DIR>/{split}/images/")
    df = pd.read_csv(csv_path)
    return _attach_derived(df, img_dir)


def load_validation(which: str = "dev") -> pd.DataFrame:
    """자체 BBQ 검증셋 로드. which ∈ {'smoke','dev','full','hard'}.

    검증셋 컬럼: sample_id, context, question, answers, label, is_ambiguous,
    context_condition, question_polarity, category, unknown_index_gt.
    이미지가 없으므로 image_abspath 는 None(텍스트 전용 평가와 동일 조건).
    """
    paths = {"smoke": config.BBQ_SMOKE, "dev": config.BBQ_DEV, "full": config.BBQ_FULL,
             "hard": config.BBQ_HARD, "val": config.BBQ_VAL}
    assert which in paths, f"unknown validation set: {which}"
    csv_path = paths[which]
    if not Path(csv_path).exists():
        raise FileNotFoundError(
            f"검증셋 없음: {csv_path}. (full 은 용량 문제로 미동봉일 수 있음 → dev 사용)")
    df = pd.read_csv(csv_path)
    # is_ambiguous 가 문자열로 들어오는 경우 방어적 캐스팅.
    if df["is_ambiguous"].dtype != bool:
        df["is_ambiguous"] = df["is_ambiguous"].astype(str).str.lower().isin(
            ["true", "1", "yes"])
    return _attach_derived(df, img_dir=None)
