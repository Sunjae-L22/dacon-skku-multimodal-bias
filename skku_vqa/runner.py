"""핵심 추론 루프 — eval(검증)과 inference(제출)가 공유한다(DRY).

프롬프트 빌드 → 백엔드 생성 → 텍스트 파싱 → (인덱스, parse_ok, raw) 수집.
단건(generate) 경로와 배치(generate_batch) 경로를 모두 지원하며, 결과는 항상
입력 DataFrame 행 순서를 보존한다.
"""
from __future__ import annotations

from typing import Optional

import pandas as pd

from . import prompts as prompts_mod
from . import parsing

try:
    from tqdm.auto import tqdm
except Exception:                     # tqdm 미설치 시 무동작 래퍼
    def tqdm(it=None, **k):
        return it if it is not None else iter(())


def _rows_to_inputs(df: pd.DataFrame, prompt_version: str, use_image: bool):
    """행 → (messages, options, unknown_index, image_path) 튜플 리스트."""
    msgs, opts, unks, imgs = [], [], [], []
    for _, row in df.iterrows():
        options = list(row["options"])
        opts.append(options)
        unks.append(row.get("unknown_index"))
        msgs.append(prompts_mod.build_messages(
            row["context"], row["question"], options, prompt_version=prompt_version))
        imgs.append(row.get("image_abspath") if use_image else None)
    return msgs, opts, unks, imgs


def predict_dataframe(df: pd.DataFrame,
                      backend,
                      prompt_version: str = "v2",
                      use_image: bool = False,
                      batch: bool = False,
                      desc: str = "infer",
                      show_progress: bool = True) -> dict:
    """df(options/unknown_index 보유)에 대해 추론.

    batch=True 이고 backend 에 generate_batch 가 있으면 전체를 한 번에 배치 처리
    (vLLM 연속배치 권장 경로). 아니면 단건 루프(tqdm ETA 표시).

    반환: {preds, raw, parse_ok_rate, n_unparsed}.
    """
    msgs, opts, unks, imgs = _rows_to_inputs(df, prompt_version, use_image)
    n = len(msgs)
    raws: list[str] = [""] * n

    if batch and hasattr(backend, "generate_batch"):
        image_paths = imgs if use_image else None
        raws = backend.generate_batch(msgs, image_paths)
    else:
        it = range(n)
        if show_progress:
            it = tqdm(it, total=n, desc=desc, unit="개")
        for i in it:
            raws[i] = backend.generate(msgs[i], image_path=imgs[i] if use_image else None)

    preds, oks = [], []
    for i in range(n):
        idx, ok = parsing.parse_answer(raws[i], opts[i], unks[i])
        preds.append(idx)
        oks.append(ok)

    return {
        "preds": preds,
        "oks": oks,                       # 행별 파싱 성공 여부(청크 집계·진단용)
        "raw": raws,
        "parse_ok_rate": sum(oks) / n if n else 0.0,
        "n_unparsed": sum(1 for o in oks if not o),
    }
