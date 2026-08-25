"""경로·상수·실행 설정. 모든 모듈이 여기서 경로와 상수를 가져온다.

재현성을 위해 절대경로 하드코딩을 피하고, 데이터 위치는 환경변수
``SKKU_DATA_DIR`` 로 덮어쓸 수 있게 둔다(기준 평가 환경에서 운영진이 다른
경로를 줄 수 있으므로). 실행 파라미터는 ``RunConfig`` 데이터클래스로 한곳에
모아 제출물과 함께 manifest 로 기록한다.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

# --- 경로 -------------------------------------------------------------------
# pipeline/skku_vqa/config.py  →  parents[1] = pipeline/ ,  parents[2] = 프로젝트 루트
PKG_ROOT = Path(__file__).resolve().parent          # .../pipeline/skku_vqa
PIPELINE_ROOT = PKG_ROOT.parent                     # .../pipeline
PROJECT_ROOT = PIPELINE_ROOT.parent                 # .../2026 ... 챌린지

# 데이터 폴더: 기본은 <프로젝트루트>/open, 환경변수로 덮어쓰기 가능.
DATA_DIR = Path(os.environ.get("SKKU_DATA_DIR", PROJECT_ROOT / "open"))

TRAIN_CSV = DATA_DIR / "train" / "train.csv"
TEST_CSV = DATA_DIR / "test" / "test.csv"
SAMPLE_SUBMISSION = DATA_DIR / "sample_submission.csv"
TRAIN_IMG_DIR = DATA_DIR / "train" / "images"
TEST_IMG_DIR = DATA_DIR / "test" / "images"

# 자체 검증셋(공개 BBQ 기반, 라벨 포함). pipeline/validation 에 동봉.
VALIDATION_DIR = PIPELINE_ROOT / "validation"
BBQ_SMOKE = VALIDATION_DIR / "bbq_smoke.csv"
BBQ_DEV = VALIDATION_DIR / "bbq_dev.csv"
BBQ_FULL = VALIDATION_DIR / "bbq_full.csv"   # 옵션(용량 큼) — 없으면 dev 사용
# hard: dev 오류 전수 + 회귀감시용 정답 표본(프롬프트 A/B 고속 측정, scripts/make_hardset.py 생성)
BBQ_HARD = VALIDATION_DIR / "bbq_hard.csv"
# val: LoRA 체크포인트 선택용 quad-분리 홀드아웃(train/build_train_data.py 생성)
BBQ_VAL = VALIDATION_DIR / "bbq_val.csv"

OUTPUT_DIR = PIPELINE_ROOT / "outputs"

N_OPTIONS = 3  # 전 샘플 3지선다(고정)

# --- "모름/판단불가(abstain)" 옵션 의미 식별 사전 ---------------------------
# 평가셋은 표현(~10종)·위치(0/1/2)를 모두 랜덤화해 규칙기반 탐지를 막아 둠.
# 아래는 소문자 부분일치(substring)용. 정답 결정이 아니라 파싱 폴백·진단용.
# (solution/src/config.py 에서 전 샘플 100% 커버·정확히 1개 매칭으로 검증된 사전.)
UNKNOWN_LEXICON = [
    "cannot be determined",
    "can't be determined",
    "cannot answer",
    "can't answer",
    "cannot be answered",
    "not answerable",
    "not enough info",          # "not enough information" 도 이 부분문자열로 매칭
    "not enough",
    "undetermined",
    "not known",
    "unknown",
    "cannot tell",
    "can't tell",
    "impossible to tell",
    "not specified",
    "no answer",
    "not determinable",
]

# --- 기본 모델 (규칙: 2026-06-01 이전 공개 + 오픈 라이선스) ------------------
# A6000 최종 제출용(bf16, vLLM/Transformers):
DEFAULT_MODEL_A6000 = "Qwen/Qwen3-VL-8B-Instruct"            # 2025-10-15, Apache-2.0
# 로컬 M4 개발용(4bit MLX, 같은 모델 변환 → 결과 전이 양호):
DEFAULT_MODEL_MLX = "mlx-community/Qwen3-VL-8B-Instruct-4bit"
# 모델 용량 업그레이드(합법: 2025-10-04 공개, Apache-2.0). MoE active-3B라 M4 24GB
# 4bit 추론 가능(qwen3_vl_moe 지원 확인). A6000 채점은 FP8/AWQ 로 별도.
MODEL_MLX_30B = "mlx-community/Qwen3-VL-30B-A3B-Instruct-4bit"
MODEL_A6000_30B = "Qwen/Qwen3-VL-30B-A3B-Instruct"   # bf16 원본 → FP8/AWQ 로 서빙


@dataclass
class RunConfig:
    """한 번의 추론/평가 실행을 완전히 기술하는 설정. 제출물 옆 manifest 로 저장.

    재현성 규칙: 이 객체 + 코드 + 라이브러리 버전이면 오차 내 점수 복원 가능.
    """
    # 백엔드 스펙: "mock:always_unknown" | "mlx[:model_id]" |
    #              "transformers:<hf_id>" | "vllm:<hf_id>"
    backend: str = "mock:always_unknown"
    model_id: Optional[str] = None          # 실제 로드된 모델 가중치(manifest 기록·재현성)
    prompt_version: str = "v2"              # "v1"(원본) | "v2"(과소기권 교정, 기본)
    use_image: bool = False                 # 1차는 텍스트 전용(이미지 미사용)
    max_new_tokens: int = 256               # 128 은 일부 출력 잘림 → 256
    # greedy(do_sample=False) = 재현성. seed 는 샘플링/배치 결정성용.
    do_sample: bool = False
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 0
    seed: int = 1234
    batch_mode: bool = False                # generate_batch(연속배치) 경로 사용 여부
    adapter_path: Optional[str] = None      # LoRA 어댑터(학습 트랙 산출물, MLX 전용)

    def metadata(self) -> dict:
        """manifest 기록용 직렬화."""
        return asdict(self)


def env_report() -> dict:
    """재현성 manifest 에 박을 환경 스냅샷(라이브러리 버전 등)."""
    import platform
    import sys

    def _ver(mod: str) -> Optional[str]:
        try:
            m = __import__(mod)
            return getattr(m, "__version__", "unknown")
        except Exception:
            return None

    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": _ver("torch"),
        "transformers": _ver("transformers"),
        "vllm": _ver("vllm"),
        "mlx_vlm": _ver("mlx_vlm"),
        "pandas": _ver("pandas"),
        "data_dir": str(DATA_DIR),
    }
