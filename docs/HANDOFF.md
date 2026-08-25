# 인수인계 (HANDOFF) — 2026 성균관대 멀티모달 AI 챌린지 (DACON 236722)

> 이 문서 한 장으로 집(3060 Ti)·새 세션에서 전체 파이프라인을 이어받아 돌릴 수 있게
> 정리한 **단일 기준 문서**. 마지막 갱신: 2026-06-11.

---

## 0. 지금 당장 할 일 (우선순위)

1. **DACON에 `outputs/submission_reparsed.csv` 제출** — 무위험 floor 굳히기(v2+파서 2건). (5분)
2. **8B v3b 풀런 → `outputs/submission.csv` → DACON 제출** = 우리의 진짜 첫 climb
   (v3b는 검증된 최강 무기인데 아직 LB에 안 올라가 있음). 두 경로 중 편한 쪽:
   - **Lightning AI(무료 24GB, 맥북 브라우저로 지금)** ⭐ — [`docs/RUN_ON_LIGHTNING.md`](RUN_ON_LIGHTNING.md). vLLM, ~1h.
   - **집 3060 Ti(8GB, Windows)** — [`docs/RUN_ON_3060TI.md`](RUN_ON_3060TI.md). transformers 4bit, ~1-2h.
3. **30B-A3B**: Lightning AI(24GB)에서 8B 다음에 도전(§5, AWQ+vLLM, bleeding-edge).

---

## 1. 대회 개요

- **과제**: BBQ(Bias Benchmark for QA)의 멀티모달 포트. 객관식 3지선다(라벨 0/1/2).
  각 문항은 context+question+선택지 3개(정확히 1개가 '모름/판단불가'류).
  - **ambiguous**: 근거 불충분 → 정답은 '모름' 옵션.
  - **disambiguated**: 근거 명시 → 정답은 특정 인물 옵션.
- **평가**: `Balanced Accuracy = (Acc_ambiguous + Acc_disambiguated) / 2`. 두 그룹 동일 가중.
- **데이터**: test 8,500(라벨 비공개) + 포맷 샘플 1개. 학습데이터는 참가자가 직접 구축.
  test에 이미지 동봉(8500장)이나 **답 기여는 텍스트에 있음**(아래 §6).
- **일정**: 마감 **6/29**, 코드검증 7/2~7/10, 시상 7/14. 일 5회 제출. Private(40%)+코드검증.
- **규칙 전문**: [`docs/COMPETITION_RULES.md`](COMPETITION_RULES.md). 하드룰 요약:
  - 2026-06-01 이전 공개 오픈가중치만 / 외부 API 금지(로컬만) / 최종답=LLM 생성텍스트
    (단순 다수결·룰매핑 금지) / 0.5s/샘플 **권장**(강제 아님, A6000 채점) /
    평가셋 패턴 분석으로 학습데이터·프롬프트 생성 = **Data Leakage 금지** / **공개 BBQ 학습은 운영진 허용 확인**.

---

## 2. 현재 상태

- **LB 0.99025 = 약 56등** (336명). 1위 1.0 = 5팀(BBQ 정답매칭 의심 → 코드검증 탈락 위험).
- ⚠️ **현재 제출본은 v2 프롬프트.** 우리의 검증된 최강 무기 **v3b는 아직 LB에 안 올라가 있음**
  (dev/hard만 측정, test 풀런 미실행). → 56등은 2군으로 싸운 결과.
- **목표**: 1.0이 아니라 **견고한 0.995~0.998** + 코드검증 생존(재현 가능·일반화). public 과적합 회피.

### 보유 산출물
| 파일 | 내용 |
|---|---|
| `outputs/submission_reparsed.csv` | v2+파서수정(2건 flip). **즉시 제출용 floor** |
| `outputs/eval_dev_v2_pred.csv` / `_errors.csv` | dev 1988 v2 결과(balanced 0.9698) |
| `validation/bbq_hard.csv` (190) | 프롬프트 A/B 고속 flip 측정셋 |
| `validation/bbq_dev.csv` (1988) / `bbq_val.csv` (2000) | 검증 / LoRA 체크포인트 선택 |
| `train/data/sft_dir/train.jsonl` (12k) | LoRA SFT 데이터(공개BBQ, dev제외) |
| `~/.cache/huggingface/.../Qwen3-VL-30B-A3B-Instruct-4bit` (17G) | 30B(클라우드용 보관) |

---

## 3. 컴퓨트 맵 — 어디서 뭘 돌리나 (★중요)

| 머신 | 스펙 | 돌릴 수 있는 것 | 못 하는 것 |
|---|---|---|---|
| **M4 MacBook** | 24GB 통합RAM | 8B-4bit MLX(개발·진단), 8B LoRA 학습 | ❌ **30B-4bit = OOM으로 시스템 크래시**(실측 2회 강종) |
| **집 3060 Ti** | 8GB VRAM, Windows | **8B-4bit CUDA 배치추론(빠름~1-2h)** + 재현성 박스 | ❌ 30B(17GB) 안 들어감 |
| **Lightning AI(무료)** | L4/A10G **24GB**, 22h/월 | **8B(쉬움) + 30B-A3B AWQ(vLLM)**, 최종 재현 CSV | A10G=Ampere=A6000. [가이드](RUN_ON_LIGHTNING.md) |
| **Kaggle(무료)** | 2×T4 32GB, 30h/주 | 8B, 30B(tensor-parallel) | T4=Turing(재현성 살짝 덜) |

- **M4에서 30B 절대 로드 금지** — 24GB 초과로 머신이 강제종료됨.
- **3060 Ti는 Ampere(sm86) = 채점 A6000과 같은 아키** → 여기 결과가 A6000에서 재현됨(룰⑦ 해결).

---

## 4. 즉시 실행: 3060 Ti에서 8B v3b 풀런

상세: [`docs/RUN_ON_3060TI.md`](RUN_ON_3060TI.md). 요약:
1. Windows에 `pipeline\` 폴더 + `open\test\test.csv`만 복사(이미지 불필요 — 텍스트 전용).
2. `pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124`
   + `pip install -r requirements-win-3060ti.txt`
3. 실행:
   ```powershell
   $env:SKKU_DATA_DIR = "C:\skku\open"
   python inference\run_inference.py --backend transformers:Qwen/Qwen3-VL-8B-Instruct `
     --prompt v3b --load-in-4bit --batch-size 4 --max-new 160
   ```
4. `outputs\submission.csv` → Mac으로 회수 → DACON 제출.
   - ⚠️ git-source transformers 금지(안정 4.57만). 채점 재현의 핵심.

---

## 5. 전략 로드맵

공유된 0.99633 솔루션 분석 결과(공유자 본인 회고 포함):
- 그들의 +0.006 출처: **모델 용량 27B (55~70%)** > 이미지(15~30%, 그들 27B 한정) > 프롬프트(~0, 우리 v3b와 수렴) > 2-pass verifier(**마이너스**, 본인도 "안 한 게 나았다").
- 그들 약점: 오프라인 검증 없이 public 직튜닝(과적합), git-source transformers(채점 재현 실패 위험 → 0.99633 무효 가능).

### 우리 경로 (우선순위)
1. **v3b 8B 제출** (3060 Ti) — disambiguated 과기권 ~296건 회수. +0.002~0.004. **즉시 climb.**
2. **30B-A3B 업그레이드 (클라우드 필수)** — 그들 우위의 본체(모델용량)를 합법 회수.
   - 모델: `Qwen/Qwen3-VL-30B-A3B-Instruct` (2025-10-04, Apache-2.0, MoE active-3B).
   - 클라우드 24GB+ GPU 렌트(vast.ai/runpod, ~$0.3-0.5/h). vLLM AWQ/FP8로 빠름.
   - **오프라인 Balanced-Acc 하니스(dev 1988)에서 8B-v3b를 이길 때만 제출** (public 직튜닝 금지).
3. **최종 채점 CSV는 A6000(클라우드)에서 생성** — 룰⑦ 재현성(M4-4bit↔A6000 불일치 = 0점 방지).
   - 3060 Ti(Ampere)에서 만든 8B CSV는 A6000 재현 OK → 8B 경로는 클라우드 불필요할 수 있음.
4. **LoRA = 백업** — 8B 텍스트SFT(천장 ~0.996), 30B 성공시 불필요. 30B LoRA는 RAM 초과로 불가.

### 기각된 것
- **이미지 사용**: 실측(`scripts/diag_image_ablation.py`) 8B에서 이득0/손해3. 분석 ~100:1 손해. **텍스트 전용 유지.**
- **2-pass verifier**: 공유자 본인이 점수 깎였다고 확인. 1-pass greedy 유지.
- **1.0 추격**: BBQ 정답매칭은 코드검증 탈락 위험. 견고한 0.995~0.998 목표.

---

## 6. 핵심 발견 (왜 이렇게 결정했나)

- **최대 약점 = disambiguated 과기권.** test의 32.7%가 멀티모달 템플릿("The image shows …
  including X and Y people. A X person did Z" + 선택지 "The X person in the red shirt").
  근거가 텍스트에 명시됐는데 v2가 "어느 위치 사람인지 모름"이라며 기권(~296건). dev엔 이
  템플릿이 0건이라 **dev 게이트가 이 약점을 못 봄**(과거 시간낭비 원인).
- **v3b가 텍스트만으로 이걸 12/12 교정**(`scripts/diag_visual.py`). 집단명이 context→option
  매핑, 시각묘사는 곁다리. **이미지 불필요.**
- **이미지는 우리 8B엔 독**: 텍스트로 맞히던 걸 오답/기권으로 망침(3/16). 모호 케이스엔
  외형 편향 주입 위험(4907건 중 1178건이 시각묘사 선택지).
- **우리가 놓쳤던 것**: ① v3b를 LB에 안 올림 ② dev 게이트의 맹점 ③ "30B 불가"를 추론까지
  오해(실제론 추론도 24GB OOM 크래시 — 더 나쁜 쪽으로 맞음).

---

## 7. 파이프라인 구조 (key files)

```
pipeline/
├─ skku_vqa/
│  ├─ config.py            경로·상수·UNKNOWN_LEXICON·RunConfig·모델ID(8B/30B)
│  ├─ data.py              CSV 로드 + 옵션파싱 + unknown 식별 (load_split/load_validation)
│  ├─ prompts.py           v0(LoRA단축)·v1·v2·**v3a·v3b** 시스템프롬프트, build_messages
│  ├─ parsing.py           LLM텍스트→인덱스 (Final answer 우선 + 'answer is X' 폴백)
│  ├─ metrics.py           Balanced Accuracy
│  ├─ runner.py            공유 추론 루프(단건/배치)
│  ├─ submission.py        제출 무결성 검증
│  └─ backends/
│     ├─ mlx_backend.py        M4 4bit(개발) — adapter_path(LoRA) 지원
│     ├─ transformers_backend.py  ★CUDA 4bit+배치(3060 Ti/A6000) — 재현성 경로
│     └─ vllm_backend.py       A6000 연속배치(0.5s) — CUDA/Linux
├─ inference/run_inference.py   test→submission.csv (+manifest). --prompt --load-in-4bit --batch-size
├─ eval/run_eval.py             검증셋 Balanced Acc (+ analyze_errors)
├─ scripts/
│  ├─ run_chunked.py        재개가능 청크러너(M4 장시간용, --prompt/--adapter-path/--target)
│  ├─ make_hardset.py       bbq_hard.csv 생성
│  ├─ compare_runs.py       두 예측 flip(고침/회귀) 비교
│  ├─ reparse_test.py       저장 청크 재파싱→submission_reparsed.csv
│  ├─ diag_visual.py        시각 과기권 프롬프트 진단(SKKU_MLX_MODEL로 모델교체)
│  └─ diag_image_ablation.py  텍스트 vs 이미지 진단
├─ train/                   LoRA 트랙(build_train_data.py, mlx_lora_wrapper.py) — 8B 백업
├─ docs/  COMPETITION_RULES.md · RUN_ON_3060TI.md · HANDOFF.md(이 문서)
└─ requirements-{eval,mac,a6000,win-3060ti}.txt · tests/test_pipeline.py(10/10)
```

검증 명령(어디서나, GPU 불필요): `python tests/test_pipeline.py`

---

## 8. 열린 리스크 & 결정 필요

- **[최대] A6000 재현성(룰⑦)**: 최종 CSV는 채점환경과 같은 CUDA로 생성해야. 8B는 3060 Ti로
  해결, 30B는 클라우드 A6000 필요. **클라우드 접근계획 미확정 = 블로커.**
- **30B 비용/셋업**: 클라우드 GPU 렌트 의사결정 필요. 8B-v3b만으로 충분한지 LB 보고 판단.
- **public 과적합**: 오프라인 하니스를 단일 판정기준으로(공유자의 함정 회피).
- **LoRA 룰⑧ gray zone**: raw-BBQ 재구성 합법성 — 운영진 허가는 받았으나 톡board 재확인 권장.
- **증빙 보관**: 운영진의 "공개BBQ 학습 허용" 답변 캡처 보관(코드검증 소명용).
- **최종 채점파일 1개 선택**: DACON 제출 UI에서 명시 선택(잊지 말 것).

---

## 9. 제출 전략

- 일 5회 제한 → 1-2회만 실제 신규 제출(floor 굳히기 또는 public-private 갭 정보 구매용).
  무작위 변형 난사 금지.
- floor(0.99025) 항상 보존: 새 제출이 floor 미만이면 v2 CSV 유지.
- 판정 기준 = **오프라인 Balanced-Acc 하니스**(dev 1988) > public LB. 1.0(5팀)은 추격 안 함.
- 최종 채점파일 = 하니스 최고 + A6000 재현 검증 통과한 CSV.
