# 제출 기록 (Submission Log)

> ★★ **최종 채점파일 = `2026-06-14_32B_v3b_LB0.99875.csv` (4bit)** — DACON 제출 UI에서 이걸로 선택.
> 파일명 = `날짜_방법_LB점수.csv`. 원본 보존(덮어쓰지 말 것).

| 날짜 | 파일 | 방법 | Public LB | 등수 | 메모 |
|---|---|---|---|---|---|
| 2026-06-11 | `2026-06-11_reparsed_v2_LB0.99025.csv` | 8B MLX 4bit · v2 · 파서수정 | 0.99025 | 56 | 초기 floor |
| 2026-06-14 | `2026-06-14_v3b_8B_LB0.99533.csv` | 8B(bf16, vLLM/L4) · **v3b** · 텍스트전용 | 0.99533 | 41 | 프롬프트 교정 |
| 2026-06-14 | **`2026-06-14_32B_v3b_LB0.99875.csv`** | **Qwen3-VL-32B(AWQ-4bit, vLLM/L40S)** · v3b · greedy | **0.99875** | **14** | ★★**최종 채택**. dev 0.9904 |
| 2026-06-15 | `2026-06-15_32B_8bit_v3b_LB0.99775.csv` | 위와 동일 모델 **8bit** | 0.99775 | — | ❌ 미채택(아래 교훈) |

## ★ 핵심 교훈: dev–LB 괴리 (과적합 함정)
- 8bit는 **dev 0.9925 > 4bit 0.9904**였지만 **LB 0.99775 < 4bit 0.99875**. 4bit 대비 162건 flip 중 순효과 −0.001.
- 원인: dev(공개 BBQ, 텍스트전용)는 test의 멀티모달 템플릿(32.7%)을 0건 포함 → ±0.001 정밀도에선 **dev가 LB를 예측 못 함.**
- 결론: **dev 향상이 LB로 전이된다는 보장 없음.** LoRA·앙상블 등 dev-최적화 레버는 같은 함정(LB 회귀) 위험 → 추가 도박 중단. 4bit 0.99875 굳히기.

## ★ 2차 평가(코드검증, 7/2 제출 · 7/10 검증) 준비 메모
- 모델: `Qwen/Qwen3-VL-32B-Instruct` (2025-10-21, Apache-2.0 = 룰④ 합법). 추론 4bit = `cyankiwi/Qwen3-VL-32B-Instruct-AWQ-4bit`(compressed-tensors).
- 제출 코드: `inference/run_inference.py --backend vllm:cyankiwi/Qwen3-VL-32B-Instruct-AWQ-4bit --prompt v3b --max-model-len 2048 --max-new 160` (텍스트전용 greedy).
- 환경: vLLM compressed-tensors Marlin, 채점 A6000(Ampere sm80) 재현 OK. **★FP8 금지**(A6000 미지원), **git-source transformers 금지**(안정 4.57만).
- 속도: 106ms/샘플 → 0.5초/70분 룰 충족.
- 합법성: 외부 데이터/매칭 無, 학습 無(zero-shot), 최종답=LLM 생성텍스트(parsing.py 추출). 룰 전조항 준수.

## 판정 원칙
- 1.0 상위 10팀 = BBQ 매칭 의심 → 코드검증 탈락 위험. 우리는 **견고·재현가능한 0.99875**로 합법 상위권 + 검증 생존.
- 오프라인 dev는 ±0.001 미만에선 신뢰 불가 → 신규 제출은 신중히(public 과적합 금지).
