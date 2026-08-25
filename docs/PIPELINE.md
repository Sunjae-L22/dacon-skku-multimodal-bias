# pipeline/ — 2026 성균관대학교 멀티모달 AI 챌린지 추론 파이프라인 (그린필드 v2)

BBQ(편향 QA)의 멀티모달 확장 객관식 VQA 과제를 푸는 **재현 가능·규칙 준수** 추론
파이프라인. `solution/`(현 LB 0.9715 작동본)과 독립된 새 구현이며, 그쪽 코드를
복사하지 않고 검증된 *사실*만 반영했다.

> **과제 한 줄**: 근거가 충분하면 그 인물을, 부족하면 '모름'을 고르는 **캘리브레이션 게임**.
> 평가: `Balanced Accuracy = (Acc_ambiguous + Acc_disambiguated) / 2`. 두 그룹 동일 가중.

## 핵심 설계 결정 (데이터로 검증됨)
- **텍스트 전용.** test 이미지 30장 표본조사 결과 답 기여 0/30. ambiguous 정답은 '모름'
  이라 이미지가 풀면 오히려 편향(회귀). → 1차는 이미지 미사용(`--use-image` 끔).
- **프롬프트 v2 (과소기권 교정).** 병목은 *disambiguated 과소기권*(근거 있는데 '모름'
  선택). v2 는 "명시 사실은 근거로 사용 + 1-step 추론 허용, 진짜 무정보일 때만 모름,
  금지는 context 없는 그룹 일반화로 한정". ambiguous(≈1.0)는 유지.
- **unknown 옵션은 의미식별(부분문자열)** 로 찾되 **정답 결정엔 안 씀** — 파싱 폴백·진단용.
  최종 답은 항상 LLM 생성 텍스트에서 추출.

## 구조
```
pipeline/
├─ skku_vqa/                  # 공통 패키지(설치 불필요, 진입 스크립트가 sys.path 추가)
│  ├─ config.py               # 경로(SKKU_DATA_DIR override)·상수·UNKNOWN_LEXICON·RunConfig
│  ├─ data.py                 # CSV/검증셋 로드 + unknown 의미식별 + 이미지 경로
│  ├─ prompts.py              # v1/v2 시스템 프롬프트, build_messages (A/B 교체)
│  ├─ parsing.py              # LLM 텍스트 → 옵션 인덱스(Final answer 우선 + 폴백)
│  ├─ metrics.py              # Balanced Accuracy
│  ├─ runner.py               # 공유 추론 루프(단건/배치), eval·inference 가 재사용
│  ├─ submission.py           # 제출 무결성 검증
│  └─ backends/               # generate(messages, image_path) 통일 인터페이스
│     ├─ mock.py              #   GPU 불필요(배관/테스트)
│     ├─ mlx_backend.py       #   로컬 M4(개발, 4bit)
│     ├─ vllm_backend.py      #   ★A6000 최종 제출(연속배치 → 0.5s/샘플)
│     └─ transformers_backend.py  # A6000 단건 폴백/대조
├─ inference/run_inference.py # test.csv → submission.csv (+manifest, 속도 로깅)  [추론 전용]
├─ train/                     # LoRA 학습 트랙 (inference 와 분리)  [현재 무학습]
│  ├─ build_train_data.py     #   공개 BBQ → SFT JSONL (leakage 차단)
│  └─ train_lora.py           #   LoRA 진입점(mlx/peft)
├─ eval/                      # 검증 하니스
│  ├─ run_eval.py             #   BBQ smoke/dev → Balanced Accuracy
│  └─ analyze_errors.py       #   오류 유형(과소기권 등)·카테고리 분해
├─ validation/                # 자체 BBQ 검증셋(라벨 포함, 재사용)
│  ├─ bbq_smoke.csv (204)  ├─ bbq_dev.csv (1988)
├─ configs/default.yaml       # 최종 제출 권장 설정(문서화)
├─ tests/test_pipeline.py     # 단위/통합(Mock, GPU 불필요)
└─ requirements-{eval,mac,a6000}.txt
```

## 설치
| 환경 | 파일 | 용도 |
|---|---|---|
| 분석·배관 (어디서나) | `requirements-eval.txt` | Mock 로 전체 흐름 GPU 없이 |
| 로컬 M4 | `requirements-mac.txt` | MLX 4bit, 프롬프트 반복 |
| **A6000 최종** | `requirements-a6000.txt` | vLLM, 0.5s/샘플, 재현 |

> ⚠️ **MLX(로컬)는 반드시 전용 venv 에 설치.** `mlx-vlm`(현행 0.6.2)이 transformers 5.x 를
> 끌어와 공유 conda base(다른 프로젝트의 transformers<5·numpy<2 의존성)를 깨뜨린다.
> ```bash
> cd pipeline && python3 -m venv .venv
> .venv/bin/python -m pip install -r requirements-mac.txt
> .venv/bin/python eval/run_eval.py --backend mlx --set smoke --limit 40
> ```

데이터 경로는 기본 `<프로젝트루트>/open`. 다르면 `export SKKU_DATA_DIR=/path/to/open`.

## 빠른 시작
```bash
# 0) 배관 점검 (GPU 불필요) — 테스트 + Mock 추론/검증
python tests/test_pipeline.py
python inference/run_inference.py --backend mock:always_unknown --limit 50
python eval/run_eval.py --backend mock:always_unknown --set smoke

# 1) 로컬 M4 개발 (MLX) — 검증셋 점수 + 오류분석
python eval/run_eval.py --backend mlx --set dev --save

# 2) A6000 최종 제출 (vLLM, 텍스트 전용, v2, 8500건)
python inference/run_inference.py --backend vllm:Qwen/Qwen3-VL-8B-Instruct --batch
#    → outputs/submission.csv + submission_manifest.json(설정·환경·ms/샘플·규칙충족)
```

## ★ 대회 규칙 준수 체크리스트
- [x] **train / inference 코드 분리** — `train/` vs `inference/run_inference.py`, 상호 import 없음.
- [x] **최종 답 = LLM 생성 텍스트** — `parsing.parse_answer` 가 모델 출력에서 추출.
      unknown 의미식별은 파싱 폴백·진단용일 뿐 정답 결정 수단 아님.
- [x] **추론 0.5초/샘플 (A6000)** — `vllm_backend` 연속배치. `run_inference` 가 ms/샘플
      실측을 manifest 에 기록(`meets_0.5s_rule`).
- [x] **2026-06-01 이전 공개 오픈 가중치** — Qwen3-VL-8B-Instruct (2025-10-15, Apache-2.0).
- [x] **Python 전용 / API 금지** — 로컬 가중치만 로드(vllm/mlx/transformers), 오프라인.
- [x] **Data Leakage 금지** — 학습 원천은 공개 BBQ 만, `build_train_data.py` 가 평가셋 거부.
- [x] **재현성** — `RunConfig` + `env_report()` 를 manifest 로 저장. greedy(temperature=0), seed 고정.
- [x] **제출 무결성** — `submission.validate_submission`(행수·정렬·라벨범위·중복) 통과해야 저장.

## 검증 상태
- 단위/통합 테스트 **9/9 통과**(파서·unknown식별·metric·runner·제출검증·비정수라벨거부·NaN안전).
- **실데이터 end-to-end** (Mock): test.csv 8500건 → submission 무결성 통과·sample_submission
  일치; BBQ dev 1988건 → always_unknown 이 balanced **0.5000**(ambiguous 1.0 / disambig 0.0)로
  산식·오류분류 정확 동작 확인.
- **실모델 MLX (Qwen3-VL-8B-4bit, .venv)**: BBQ smoke 40건(ambig 20 + disambig 20) →
  balanced **1.0000**, parse_ok 100%, ~6s/샘플. 라이브 모델 경로(로드→생성→파싱→채점) 확인.
  (smoke 는 쉬운 서브셋 — 풀 dev 의 disambiguated 캘리브레이션 수치는 별도 측정.)
- **vLLM(A6000)** 은 CUDA 전용이라 이 M4 에선 미측정 — A6000 에서
  `inference/run_inference.py --backend vllm:Qwen/Qwen3-VL-8B-Instruct` 로 ms/샘플 실측
  (manifest `meets_0.5s_rule`). 모델 가중치는 백엔드 `__init__` 에서만 로드(지연).
