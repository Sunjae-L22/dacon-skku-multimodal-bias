# train/ — 학습 트랙 (inference 와 물리 분리)

대회 재현성 규칙상 **학습 코드와 추론 코드는 별도 파일**이어야 한다. 이 폴더가 학습
트랙이고, 제출 추론은 [`../inference/run_inference.py`](../inference/run_inference.py) 가
담당한다. 둘은 서로를 import 하지 않는다(공통 부품만 `skku_vqa` 패키지에서 공유).

## 현재 상태: 무학습(zero-shot)
1차/베이스라인 제출은 **파인튜닝 없이** Qwen3-VL-8B + 캘리브레이션 프롬프트(v2)로 낸다.
이 폴더는 추후 캘리브레이션을 가중치에 직접 학습시키는 **선택 트랙**의 진입점이다.

## 구성
- [`build_train_data.py`](build_train_data.py) — 공개 BBQ → 챌린지 포맷 SFT(JSONL).
- [`train_lora.py`](train_lora.py) — LoRA 파인튜닝 진입점(engine: `mlx` 로컬 / `peft` A6000).

## ★ Data Leakage 규칙 (엄수)
- 학습 데이터 원천은 **공개 BBQ(CC-BY)뿐.** 대회 평가셋(`test.csv`)·Hidden 셋을
  원천으로 학습 데이터를 만들면 **실격**이다.
- `build_train_data.py` 는 입력 경로 이름에 `test`/`hidden` 이 있으면 **즉시 거부**한다.
- 자체 dev 검증셋(`validation/bbq_dev.csv`)에 든 `sample_id` 는 학습에서 제외해
  train/val 누수를 막는다(점수 신뢰성).

## 사용 예
```bash
# 1) 공개 BBQ → SFT 데이터 (dev 중복 제외)
python train/build_train_data.py --bbq-source validation/bbq_full.csv \
    --exclude-dev validation/bbq_dev.csv --out train/data/sft.jsonl

# 2) LoRA 학습 (A6000: peft / 로컬: mlx)
python train/train_lora.py --engine peft --data train/data/sft.jsonl --epochs 1
```
> `train_lora.py` 의 실제 학습 루프는 무학습 베이스라인 점수를 확정한 뒤 채운다(scaffold).
> 학습 후 adapter 와 `train_manifest.json`(seed·하이퍼파라미터·라이브러리 버전)을 저장해
> 추론에서 base 모델에 로드한다.
