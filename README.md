# 언제 답하고, 언제 "모른다"고 할 것인가

**2026 성균관대학교 멀티모달 AI Bias 챌린지 (DACON) — 최종 8위 / 263명 · 수상**

[![Award](https://img.shields.io/badge/DACON-8th%20%2F%20263-FFB800)](docs/certificate.png)
[![CI](https://github.com/Sunjae-L22/dacon-skku-multimodal-bias/actions/workflows/ci.yml/badge.svg)](https://github.com/Sunjae-L22/dacon-skku-multimodal-bias/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.12-3776AB)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

편향 벤치마크 QA에서 **모델을 학습시키지 않고 프롬프트만으로** 판단 기준을 교정해
Private Balanced Accuracy **0.9075**, 최종 **8위 / 263명**을 기록한 추론 파이프라인입니다.

> 🏅 [DACON Certificate of Excellence](docs/certificate.png) · 인증코드 `#D20260729000009` · 상금 30만원
> 📄 [2차 평가 발표자료 (14p)](docs/presentation-2026-skku-challenge.pdf) · [제출 이력](outputs/submissions/SUBMISSIONS.md) · [기술 문서](docs/PIPELINE.md)

---

## 문제

BBQ(Bias Benchmark for QA)를 멀티모달로 확장한 3지선다 과제입니다. 모든 문항에 "모름" 선택지가 정확히 하나씩 있고, 문제는 **언제 그걸 골라야 하는가**입니다.

| 문항 유형 | 상황 | 정답 |
|---|---|---|
| **ambiguous** | 맥락에 근거가 없음 | **"모름"** |
| **disambiguated** | 맥락이 특정 인물을 지목함 | 그 **인물** (주로 반고정관념) |

평가는 `Balanced Accuracy = (Acc_ambiguous + Acc_disambiguated) / 2`. 두 그룹에 동일 가중이라 **한쪽으로 치우치면 바로 손해**입니다. 전부 "모름"으로 찍으면 정확히 0.5000이 나옵니다.

주요 제약: 로컬 오픈웨이트 LLM만 사용(외부 API 금지) · 2026-06-01 이전 공개 가중치만 · 최종 답변은 반드시 LLM이 생성한 텍스트여야 함(규칙 기반 매핑 금지) · 추론 0.5초/샘플 이내.

---

## 결과

| 날짜 | 구성 | Public LB | 당시 순위 |
|---|---|---|---|
| 06-05 | Qwen3-VL-8B · 프롬프트 v1 | 0.9715 | 31 |
| 06-11 | 8B (MLX 4bit) · v2 · 파서 수정 | 0.99025 | 56 |
| 06-14 | 8B (bf16, vLLM) · **v3b** · 텍스트 전용 | 0.99533 | 41 |
| **06-14** | **Qwen3-VL-32B (AWQ-4bit, vLLM)** · v3b · greedy | **0.99875** | **14** ★ 최종 채택 |
| 06-15 | 위와 동일 모델 8bit | 0.99775 | — ❌ 미채택 |

**Private (최종) Balanced Accuracy 0.9075 → 8위 / 263명**
추론 속도 **106 ms/샘플** (규칙 상한 500ms) · 파싱 성공률 ~100% · **파인튜닝 없음 (zero-shot)**

오프라인 검증셋(공개 BBQ 1,988문항) 기준: v2 0.9698 → v3b·32B **0.9904**

---

## 핵심 의사결정 4가지

### 1. 멀티모달 대회인데 이미지를 껐다

test 이미지 30장을 직접 표본조사한 결과 **답에 기여하는 이미지가 0/30**이었습니다. 워터마크가 박힌 스톡 사진, 텍스트가 얹힌 배너, 선택지와 무관한 인물 사진뿐이었고 — ambiguous 문항의 정답은 애초에 "모름"이라 이미지가 무언가를 "읽어내면" 오히려 외모 기반 편향이 주입될 위험이 있었습니다.

ablation으로 실측했더니 이득 0건, 손해 3건. `use_image=False`로 확정했습니다.
→ [`scripts/diag_image_ablation.py`](scripts/diag_image_ablation.py)

### 2. 라벨 없는 평가셋을 "측정 가능하게" 만들었다

test 라벨이 비공개라 개선 여부를 알 수 없는 상태였습니다. 공개 BBQ를 대회 포맷으로 재구성해 **ambiguous / disambiguated를 분리 측정하는 오프라인 하니스**를 만들었습니다 (dev 1,988 · val 2,000 · hard 190 · smoke 204). 이후 모든 결정은 여기서 먼저 검증했습니다.
→ [`validation/`](validation/), [`eval/run_eval.py`](eval/run_eval.py)

### 3. 병목을 특정하고, 가중치가 아니라 프롬프트로 고쳤다

v1의 오답 124건을 유형별로 분해했습니다.

```
DISAMBIG_ABSTAIN   117건   ← 근거가 있는데 "모름"을 고름 (과소기권)
DISAMBIG_WRONG       5건
AMBIG_OVERCOMMIT     2건
```

**실점의 94%가 "과소기권" 한 가지**였습니다. ambiguous 쪽은 이미 거의 완벽했고요. v3b에서 "구별 사실(differentiating fact)"을 결정축으로 명시하고 대조 예시 3개를 넣어 이 행동만 교정했습니다. hard셋 0.750 → **0.823**.
→ [`skku_vqa/prompts.py`](skku_vqa/prompts.py)

### 4. dev가 오르는데 LB가 떨어졌을 때, 원인을 밝히고 멈췄다

8bit 양자화가 dev에서는 더 좋았지만(0.9925 > 0.9904) Public LB는 더 나빴습니다(0.99775 < 0.99875). 4bit 대비 162건이 뒤집혔고 순효과는 −0.001.

원인은 **검증셋 자체의 편향**이었습니다. dev는 공개 BBQ 기반 텍스트 전용이라 test의 멀티모달 템플릿(32.7%)을 **0건 포함**합니다. ±0.001 정밀도에서는 dev가 LB를 예측하지 못한다는 뜻이죠.

여기서 LoRA·앙상블 같은 dev-최적화 레버를 더 당기면 같은 함정에 다시 빠집니다. **추가 도박을 중단**하고 0.99875를 굳혔습니다.
→ [`outputs/submissions/SUBMISSIONS.md`](outputs/submissions/SUBMISSIONS.md)

> 참고로 Public LB 1.0을 기록한 상위 팀들은 BBQ 원본 매칭이 의심되어 코드 검증 탈락 위험이 있었습니다. "출처 불명 near-perfect"보다 **재현 가능한 상위권**을 택했고, 결과적으로 2차 코드 검증을 통과했습니다.

---

## 재현성 설계

채점 환경이 **RTX A6000 (Ampere sm80)** 이라는 점에서 역산했습니다.

- 개발은 Apple M4(MLX)와 RTX 3060 Ti, 최종 추론은 Lightning AI Studio의 **L40S / A10G** — A10G도 Ampere라 채점 환경과 아키텍처가 같습니다. 여기서 생성한 CSV는 A6000에서 재현됩니다.
- Ampere가 지원하지 않는 **FP8 금지**, API 드리프트가 있는 **git-source transformers 금지**.
- `greedy` 디코딩(temperature=0) + seed 고정으로 결정론적 실행.
- 실행할 때마다 [`submission_manifest.json`](outputs/submission_manifest.json)에 RunConfig·라이브러리 버전·ms/샘플·파싱 성공률·라벨 분포·규칙 충족 여부를 자동 기록.
- `train/`과 `inference/`는 상호 import가 없습니다. 학습 데이터 빌더는 평가셋 입력을 거부해 leakage를 차단합니다.

### 백엔드 추상화

같은 코드가 4개 환경에서 돕니다. 로컬에서 프롬프트를 반복하고 클라우드에서 제출본을 만드는 흐름을 위해서입니다.

| 백엔드 | 용도 |
|---|---|
| `mock` | GPU 없이 배관·테스트 (CI가 이걸로 돕니다) |
| `mlx` | Apple Silicon 로컬 개발 (4bit) |
| `transformers` | 단건 폴백·대조 |
| `vllm` | **최종 제출** — 연속 배치로 106ms/샘플 |

---

## 실행

```bash
# GPU 없이 전체 흐름 확인
pip install -r requirements-eval.txt
pytest tests/ -q                                                   # 10 passed
python eval/run_eval.py --backend mock:always_unknown --set smoke  # → 0.5000 (기준선)

# Docker
docker build -t skku-bias . && docker run --rm skku-bias

# 최종 제출 재현 (A6000 / L40S)
pip install -r requirements-a6000.txt
python inference/run_inference.py \
  --backend vllm:cyankiwi/Qwen3-VL-32B-Instruct-AWQ-4bit \
  --prompt v3b --max-model-len 2048 --max-new 160 --batch
```

데이터 경로는 기본 `<프로젝트루트>/open`. 다르면 `export SKKU_DATA_DIR=/path/to/open`.
(대회 제공 데이터는 라이선스상 포함하지 않았습니다. 검증셋 `validation/`은 공개 BBQ 기반이라 포함되어 있습니다.)

## 구조

```
skku_vqa/          공통 패키지 — config · data · prompts · parsing · metrics · runner · submission
  backends/          mock · mlx · transformers · vllm (통일 인터페이스)
inference/         test.csv → submission.csv (+ manifest)   [추론 전용]
train/             LoRA 트랙 (최종 미사용, leakage 차단 로직 포함)
eval/              검증 하니스 + 오류 유형 분해
validation/        공개 BBQ 기반 자체 검증셋 4종
scripts/           진단 도구 (이미지 ablation · 런 비교 · hard셋 생성 …)
tests/             단위·통합 테스트 10개 (GPU 불필요)
docs/              기술 문서 · 대회 규칙 준수 매핑 · 발표자료
```

## 문서

- [`docs/PIPELINE.md`](docs/PIPELINE.md) — 설계 결정과 구조 상세
- [`docs/COMPETITION_RULES.md`](docs/COMPETITION_RULES.md) — 대회 규칙 원문 + 조항별 준수 매핑
- [`docs/RUN_ON_LIGHTNING.md`](docs/RUN_ON_LIGHTNING.md) — 클라우드 GPU 실행 절차
- [`outputs/submissions/SUBMISSIONS.md`](outputs/submissions/SUBMISSIONS.md) — 제출 이력과 교훈

## 라이선스

MIT — [LICENSE](LICENSE) 참조. 대회 제공 데이터셋은 포함하지 않았습니다.
