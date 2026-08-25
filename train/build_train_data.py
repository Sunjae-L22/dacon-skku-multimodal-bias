#!/usr/bin/env python3
"""학습 데이터 빌더 — 공개 BBQ → 챌린지 포맷 SFT(JSONL) + val 홀드아웃 CSV.

★ Data Leakage 규칙(엄수): 학습 원천은 **공개 BBQ(CC-BY)** 뿐이다. 대회 평가셋
   (test.csv/Hidden)을 원천으로 쓰면 실격 — test/hidden 경로는 즉시 거부한다.
   (운영진이 공개 BBQ 학습 허용을 확인함 — 답변 증빙 보관.)

설계(2026-06-11 LoRA 트랙 설계 워크플로 확정안):
  · 샘플링 단위 = quad. BBQ 는 4행(ambig/disambig × neg/nonneg)이 한 컨텍스트
    베이스를 공유한다. quad 통째 샘플링이 'WHO 가 명시되는 순간만 진입'이라는
    결정 경계를 대조로 가르치고 ambig:disambig=1:1, polarity=1:1 을 구조 보장.
  · 카테고리 쿼터(quads) — 실패 구조 가중(기권 집중 카테고리 오버샘플):
      SES 450 / Race_x_SES 450 / Race_x_gender 450 / Age 350 / Race_ethnicity 250
      / Gender_identity 250 / Nationality 180 / Disability_status 180
      / Physical_appearance 180 / Religion 150 / Sexual_orientation 110  = 3,000
  · 카테고리 내부는 질문 템플릿별 층화 라운드로빈(모든 템플릿 커버, seed 고정).
  · dev 1988(=497 통quad)은 quad 단위로 제외(근접 누수 차단). hard ⊂ dev 확인.
  · val: train·dev 와 quad-분리된 500 quads(2,000행) → validation/bbq_val.csv
    (LoRA 체크포인트 선택용; run_chunked --target val 로 평가).
  · 타깃 = 'Final answer: X' 단독(~6 tok) — 파싱 100%·디코드 최소.
    시스템 프롬프트 = v0(학습·추론 비트단위 동일).

예)
  python3 train/build_train_data.py            # 기본값으로 전부 생성
  python3 train/build_train_data.py --quads 2000 --val-quads 500
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from skku_vqa import config, data as data_mod, prompts  # noqa: E402

LETTERS = ["A", "B", "C"]

# 카테고리별 quad 쿼터(설계 근거는 모듈 docstring).
QUAD_QUOTA = {
    "SES": 450, "Race_x_SES": 450, "Race_x_gender": 450, "Age": 350,
    "Race_ethnicity": 250, "Gender_identity": 250, "Nationality": 180,
    "Disability_status": 180, "Physical_appearance": 180, "Religion": 150,
    "Sexual_orientation": 110,
}
MAX_CHARS = 3500          # 비정상 길이 가드(BBQ 는 짧음 — 사실상 전건 통과)


def _refuse_if_eval_source(path: Path) -> None:
    name = path.name.lower()
    if "test" in name or "hidden" in name:
        raise SystemExit(
            f"[LEAKAGE 차단] 평가셋({path})을 학습 원천으로 쓸 수 없습니다. "
            "공개 BBQ 만 허용됩니다(대회 규칙).")


def quad_key(sample_id: str) -> tuple[str, int]:
    """'BBQ-{category}-{num}' → (category, num//4). BBQ 는 4행이 한 quad."""
    prefix, num = sample_id.rsplit("-", 1)
    return prefix, int(num) // 4


def _select_quads(cat_df: pd.DataFrame, quota: int, rng: random.Random) -> list:
    """질문 템플릿별 층화 라운드로빈으로 quota 개 quad 선택(결정적)."""
    quads = {}
    for qk, g in cat_df.groupby("quad_id"):
        tmpl = tuple(sorted(set(g["question"])))
        quads.setdefault(tmpl, []).append(qk)
    for lst in quads.values():
        lst.sort()
        rng.shuffle(lst)
    templates = sorted(quads.keys())
    rng.shuffle(templates)

    picked, i = [], 0
    while len(picked) < quota and any(quads[t] for t in templates):
        t = templates[i % len(templates)]
        if quads[t]:
            picked.append(quads[t].pop())
        i += 1
    return picked[:quota]


def main() -> None:
    default_src = config.PROJECT_ROOT / "solution" / "validation" / "bbq_full.csv"
    ap = argparse.ArgumentParser(description="공개 BBQ → 챌린지 SFT JSONL + val CSV")
    ap.add_argument("--bbq-source", default=str(default_src),
                    help="공개 BBQ CSV(검증셋과 동일 스키마)")
    ap.add_argument("--exclude-dev", default=str(config.BBQ_DEV))
    ap.add_argument("--prompt", dest="prompt_version", default="v0",
                    help="학습 시스템 프롬프트(추론과 동일해야 함, 기본 v0)")
    ap.add_argument("--quads", type=int, default=3000, help="학습 quad 수(×4=행수)")
    ap.add_argument("--val-quads", type=int, default=500,
                    help="체크포인트 선택용 val quad 수(train·dev 와 분리)")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out-dir", default="train/data/sft_dir",
                    help="HF datasets 디렉토리(train.jsonl 생성)")
    args = ap.parse_args()

    src = Path(args.bbq_source)
    _refuse_if_eval_source(src)
    df = pd.read_csv(src)
    df["quad_id"] = df["sample_id"].map(quad_key)

    # --- dev 를 quad 단위로 제외(같은 quad 의 이웃 행 근접 누수 차단) ---------
    dev = pd.read_csv(args.exclude_dev)
    dev_quads = set(dev["sample_id"].map(quad_key))
    n_dev_rows_in_quads = int(df["quad_id"].isin(dev_quads).sum())
    df = df[~df["quad_id"].isin(dev_quads)].copy()
    print(f"  dev 제외: {len(dev_quads)} quads ({n_dev_rows_in_quads}행) 차단 "
          f"→ 잔여 풀 {len(df)}행")
    if config.BBQ_HARD.exists():
        hard_ids = set(pd.read_csv(config.BBQ_HARD)["sample_id"])
        assert hard_ids <= set(dev["sample_id"]), "hard 셋이 dev 밖 샘플을 포함!"

    # --- 카테고리 쿼터 × 템플릿 층화로 train quad 선택 ------------------------
    rng = random.Random(args.seed)
    train_quads: list = []
    for cat, quota in QUAD_QUOTA.items():
        cat_df = df[df["category"] == cat]
        sel = _select_quads(cat_df, quota, rng)
        if len(sel) < quota:
            print(f"  ⚠️ {cat}: 가용 quad {len(sel)} < 쿼터 {quota}")
        train_quads += sel
    train_quads_set = set(train_quads)
    train_df = df[df["quad_id"].isin(train_quads_set)].copy()
    train_df = train_df.sample(frac=1.0, random_state=args.seed)  # 행 셔플(결정적)

    # --- val: train·dev 와 quad-분리 500 quads → validation/bbq_val.csv -------
    rest = df[~df["quad_id"].isin(train_quads_set)]
    rest_quads = sorted(set(rest["quad_id"]))
    rng.shuffle(rest_quads)
    val_quads = set(rest_quads[:args.val_quads])
    val_df = df[df["quad_id"].isin(val_quads)].drop(columns=["quad_id"])
    val_df.to_csv(config.BBQ_VAL, index=False)

    # --- SFT JSONL(단일턴 messages, 타깃 = 'Final answer: X') ------------------
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "train.jsonl"
    n, skipped = 0, 0
    with out_path.open("w", encoding="utf-8") as f:
        for _, row in train_df.iterrows():
            options = data_mod.parse_options(row["answers"])
            label = int(row["label"])
            if not (0 <= label < len(options)):
                skipped += 1
                continue
            msgs = prompts.build_messages(row["context"], row["question"], options,
                                          prompt_version=args.prompt_version)
            target = f"Final answer: {LETTERS[label]}"
            if sum(len(m["content"]) for m in msgs) + len(target) > MAX_CHARS:
                skipped += 1
                continue
            f.write(json.dumps(
                {"messages": msgs + [{"role": "assistant", "content": target}]},
                ensure_ascii=False) + "\n")
            n += 1

    sha = hashlib.sha256(out_path.read_bytes()).hexdigest()
    manifest = {
        "source": str(src), "seed": args.seed, "prompt_version": args.prompt_version,
        "quad_quota": QUAD_QUOTA, "n_train_rows": n, "n_skipped": skipped,
        "n_train_quads": len(train_quads), "n_val_rows": len(val_df),
        "n_dev_quads_excluded": len(dev_quads),
        "train_jsonl_sha256": sha, "out": str(out_path), "val_csv": str(config.BBQ_VAL),
    }
    (out_dir / "sft_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2))

    print(f"[학습 데이터] {out_path}  ({n}행, 스킵 {skipped})")
    print(f"  train 카테고리 분포: {dict(train_df['category'].value_counts())}")
    amb = train_df["is_ambiguous"].astype(bool)
    print(f"  ambig:disambig = {int(amb.sum())}:{int((~amb).sum())}")
    print(f"  label 분포: {dict(train_df['label'].value_counts().sort_index())}")
    print(f"[val 홀드아웃] {config.BBQ_VAL}  ({len(val_df)}행, train·dev 와 quad-분리)")
    print(f"[manifest] {out_dir / 'sft_manifest.json'}  sha256={sha[:12]}…")


if __name__ == "__main__":
    main()
