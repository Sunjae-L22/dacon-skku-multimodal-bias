# 내일 이어서 (2026-06-14 저녁 기준)

## 현재 위치 🏆
- **Public LB 0.99875 = 14등** (32B dense + v3b). 오늘 56등 → 14등.
- 최종 채점파일: `outputs/submissions/2026-06-14_32B_v3b_LB0.99875.csv`
  → ⚠️ **DACON 제출 UI에서 이게 "최종 선택"인지 확인** (안 했으면 내일 첫 일로).

## 오늘 검증된 사실 (내일 안 헷갈리게)
- **레버 = ① v3b 프롬프트(과기권 교정) + ② dense 32B 모델.** dev 0.9904.
- 30B-A3B(MoE)는 오히려 8B보다 **나빴음** → "크기"가 아니라 "dense"가 핵심.
- 이미지 사용은 **순손해**(텍스트 전용 유지).
- 30B/32B는 free L4(24GB) 안 들어감 → **Lightning L40S 48GB**($2.89/hr) 또는 A100 40GB 필요.
- 남은 Lightning 크레딧: 약 **$7.9**.

## 내일 재시작 방법 (Lightning)
1. Lightning Studio 열기 → **L40S(48GB) 1개**로 GPU 켜기 (Quantity 1 주의!).
2. 터미널: `cd ~/pipeline` (env 새로 export 불필요, 모델 캐시 유지됨).
3. 32B 재현/측정 명령:
   ```bash
   python eval/run_eval.py --backend vllm:cyankiwi/Qwen3-VL-32B-Instruct-AWQ-4bit \
     --set dev --prompt v3b --max-model-len 2048 --max-new 160 --batch --save
   ```
   (dev 0.9904 나오면 정상 = 기준선.)
4. 끝나면 **L40S 정지/되돌리기**로 크레딧 절약.

## 내일 후보 레버 (욕심 푸시용, dev 0.9904 넘어야 제출)
- **(B) INT8 32B** — 4bit보다 양자화 손실↓, A6000 호환 유지. cyankiwi/QuantTrio의 8bit 레포 찾아서 dev 측정. ~$2.
- **(C) LoRA on 32B** — L40S 48GB에 32B QLoRA 학습 들어갈 수도(미검증). 캘리브레이션 추가 학습. 과적합·재현성 주의.
- **(D) 앙상블** — 32B 다중 프롬프트 + LLM aggregator(룰⑥상 단순다수결 금지). 복잡·마진.
- ※ 어떤 것도 **dev(0.9904) 게이트 + 재현성(A6000=4bit only, FP8 금지)** 통과해야 채택.

## 절대 잊지 말 것
- 1.0 상위 5팀 = BBQ 매칭 의심 → 코드검증 탈락 위험. 우리 목표는 **견고·재현 가능한 0.998대**로 top-15 생존.
- 최종 제출 코드 = `inference/run_inference.py --backend vllm:cyankiwi/Qwen3-VL-32B-Instruct-AWQ-4bit --prompt v3b ...`. **git-source transformers 금지, FP8 금지.**
- 전체 맥락: `docs/HANDOFF.md` · 제출기록: `outputs/submissions/SUBMISSIONS.md` · 규칙: `docs/COMPETITION_RULES.md`.
