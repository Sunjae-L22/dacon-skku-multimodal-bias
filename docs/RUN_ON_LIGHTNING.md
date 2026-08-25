# Lightning AI(무료 24GB GPU)에서 추론 — 8B 먼저, 30B 다음

> 무료: L4/A10G **24GB**, 월 22 GPU-시간. **A10G = Ampere = 채점 A6000과 같은 아키**
> → 여기서 만든 CSV가 A6000에서 재현됨(룰⑦ 해결). 맥북 브라우저로 지금 바로 가능.
>
> **순서: ① 8B v3b(쉬움·빠름·즉시 climb) → ② 30B-A3B(큰 레버, 거친 길).**

---

## 0. Studio 생성
1. https://lightning.ai 가입(구글/이메일).
2. 새 **Studio** 생성 → 우측에서 GPU 연결: **L4** 또는 **A10G**(둘 다 24GB).
   - A10G 우선(Ampere = A6000 동일 아키, 재현성 ↑). 없으면 L4도 OK.
3. Studio = 클라우드 VS Code + 터미널. 아래는 그 터미널에서 실행.

## 1. 코드 + 데이터 올리기
- 맥에서 `pipeline/` 폴더를 zip → Studio 파일창에 드래그 업로드 → 압축해제.
- `open/test/test.csv`(작음, 이미지 불필요)도 올려서 `~/open/test/test.csv` 에 둠.
  (텍스트 전용이라 이미지 8500장은 필요 없음.)

## 2. 환경 설치 (터미널)
```bash
pip install -U vllm                 # 최신(Qwen3-VL 지원 위해 >=0.11 필요)
pip install -U transformers         # Qwen3-VL 프로세서(>=4.57)
pip install pandas pyyaml tqdm pillow
python -c "import vllm, torch; print('vllm', vllm.__version__, '| cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## 3. ★먼저: 8B v3b 풀런 (쉬움, ~수십분~1h)
```bash
cd ~/pipeline                       # 업로드/해제한 위치
export SKKU_DATA_DIR=~/open
# 24건 배관 점검(모델 로드·vLLM·파싱 OK 확인)
python inference/run_inference.py --backend vllm:Qwen/Qwen3-VL-8B-Instruct \
  --prompt v3b --max-new 160 --limit 24
# 정상이면 전체 8500 (vLLM 연속배치 자동 ON)
python inference/run_inference.py --backend vllm:Qwen/Qwen3-VL-8B-Instruct \
  --prompt v3b --max-new 160
```
→ `outputs/submission.csv` + manifest(ms/샘플·parse_ok 기록). 8B bf16(~16GB)이 24GB에 여유.
→ **이 CSV를 다운로드해서 DACON 제출 = 우리의 즉시 climb.**

## 4. 다음: 30B-A3B 풀런 (큰 레버, 거친 길)
```bash
python inference/run_inference.py \
  --backend vllm:cyankiwi/Qwen3-VL-30B-A3B-Instruct-AWQ-4bit \
  --prompt v3b --quantization awq_marlin --max-model-len 4096 \
  --gpu-mem-util 0.92 --max-new 160 --limit 24      # 먼저 24건 점검!
```
정상이면 `--limit 24` 빼고 전체. AWQ 4bit(~17GB)가 24GB에 적재.

⚠️ **30B+AWQ+vLLM 은 bleeding-edge** — 안 되면 순서대로 시도:
1. 다른 AWQ 레포로 교체: `--backend vllm:tclf90/Qwen3-VL-30B-A3B-Instruct-AWQ`
   또는 `vllm:QuantTrio/Qwen3-VL-30B-A3B-Instruct-AWQ`
2. `pip install -U vllm` 최신 재확인(MoE-VLM 지원이 버전 탐).
3. OOM 이면 `--gpu-mem-util 0.95 --max-model-len 3072`.
4. 그래도 막히면 에러 로그를 나한테 그대로 붙여줘 — 같이 디버깅.

## 5. 결과 회수 & 판정
- `outputs/submission.csv` 다운로드 → DACON 제출.
- 8B vs 30B 중 **오프라인 하니스(dev 1988)에서 이긴 쪽**을 최종 채택(public 직튜닝 금지):
  ```bash
  # dev 검증(라벨 있음)도 같은 vLLM 백엔드로 측정 가능
  python eval/run_eval.py --backend vllm:Qwen/Qwen3-VL-8B-Instruct --set dev --prompt v3b --save
  python eval/run_eval.py --backend vllm:cyankiwi/Qwen3-VL-30B-A3B-Instruct-AWQ-4bit \
    --set dev --prompt v3b --quantization awq_marlin --save
  ```
  Balanced Accuracy 높은 쪽 채택. (dev = 1988건, 몇 분.)

---

## 메모
- 무료 22h/월 — 8B 풀런 ~1h, 30B 풀런 ~1-2h, dev 측정 수분. 한도 충분.
- 세션 끊김 대비: 풀런 전 `--limit 24` 점검 필수. 끊기면 재실행(8500은 한 프로세스).
- 재현성: 최종 채점 CSV는 24GB Ampere(A10G)에서 생성한 것 사용 → A6000 재현 신뢰.
- ★git-source transformers 쓰지 말 것(안정 릴리스만) — 공유 0.99633 코드의 함정.
- 모델 전부 2025-10 공개 Apache-2.0(룰④ 합법): Qwen3-VL-8B / 30B-A3B.
