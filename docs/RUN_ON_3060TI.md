# 집 RTX 3060 Ti(8GB, Windows)에서 v3b 테스트 풀런 — 빠르게 + 재현성 확보

**왜 여기서 도나:** M4 MLX는 한 번에 1개씩(serial) 처리해 8500건에 24~56h. CUDA는
배칭이 돼서 ~2-4h. 게다가 3060 Ti는 **Ampere(sm86) = 채점 A6000과 같은 아키텍처**라
여기 결과가 A6000에서 거의 그대로 재현됨 → 룰⑦(재현성) 리스크 해소.

> 제약: VRAM 8GB라 **8B를 4bit로** 적재(됨, 빠듯). 30B-A3B(17.5GB)는 안 들어가 → M4/클라우드.
> 텍스트 전용이라 **이미지 8500장 불필요** — `test.csv`만 있으면 됨.

---

## 1. 파일 복사 (Mac → Windows)
Windows 머신에 아래만 복사(예: `C:\skku\`):
- `pipeline\` 폴더 전체 (코드)
- `open\test\test.csv` 한 파일 (→ `C:\skku\open\test\test.csv`)

## 2. 환경 설치 (PowerShell)
```powershell
# Python 3.10 권장(채점환경과 일치). conda 또는 python.org.
cd C:\skku\pipeline
python -m venv .venv-win
.\.venv-win\Scripts\Activate.ps1

# torch CUDA 휠 먼저 (CUDA 12.4)
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
# 나머지
pip install -r requirements-win-3060ti.txt
```
설치 확인:
```powershell
python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# → CUDA: True NVIDIA GeForce RTX 3060 Ti
```

## 3. 모델 받기
첫 실행 시 `Qwen/Qwen3-VL-8B-Instruct`(bf16 ~16GB)를 자동 다운로드 → **로드 시
4bit로 양자화**(VRAM ~5-6GB). 디스크 16GB 여유 필요.

## 4. 데이터 경로 + 실행
```powershell
$env:SKKU_DATA_DIR = "C:\skku\open"
# 먼저 24건으로 배관 점검(모델 로드·4bit·배치·파싱 OK 확인, 수 분)
python inference\run_inference.py --backend transformers:Qwen/Qwen3-VL-8B-Instruct `
  --prompt v3b --load-in-4bit --batch-size 4 --max-new 160 --limit 24

# 정상이면 전체 8500 (배치는 자동 ON). 8GB면 batch 4로 시작, 여유 있으면 8로.
python inference\run_inference.py --backend transformers:Qwen/Qwen3-VL-8B-Instruct `
  --prompt v3b --load-in-4bit --batch-size 4 --max-new 160
```
→ `outputs\submission.csv` + `outputs\submission_manifest.json`(속도·parse_ok 기록).

## 5. 결과 회수
`outputs\submission.csv`를 Mac으로 가져와 DACON 제출. flip 감사는 Mac에서:
```bash
python3 scripts/compare_runs.py outputs/eval_dev_v2_pred.csv <(...)   # 또는 v2 제출본과 diff
```

---

## 튜닝/문제해결
- **CUDA out of memory**: `--batch-size 2` 로 낮춰. 그래도면 `--max-new 128`.
  (8B-4bit + v3b ~1k토큰 프롬프트라 8GB는 빠듯 — batch 2~4가 안전선.)
- **너무 느림**: 배관 점검 로그의 `ms/샘플` 확인. batch를 8로 올려보고 OOM 안 나면 유지.
  3060 Ti면 batch4에서 대략 0.4~0.8s/샘플 기대 → 8500건 ~1-2h.
- **bitsandbytes 설치 오류(Windows)**: `pip install -U bitsandbytes` 최신으로. 그래도
  안 되면 WSL2 Ubuntu에서 동일 절차(vLLM도 가능) — 그땐 알려줘.
- **재현성**: 이 경로(transformers 4.57 + bitsandbytes nf4, greedy)가 곧 제출 코드.
  A6000 채점도 같은 코드로 돌아가 borderline 아이템까지 일치 → 최종 채점 CSV의
  신뢰 기반. ★git-source transformers 쓰지 말 것(공유 0.99633 코드의 함정).
