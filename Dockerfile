# GPU 없이 파이프라인 전체(추론 배관 → 제출 검증 → 오류분석)를 재현하기 위한 이미지.
# 최종 제출용 GPU 실행은 requirements-a6000.txt + vLLM 환경을 따로 씁니다(README 참조).
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# 의존성 먼저 복사 → 소스 변경 시 레이어 캐시 유지
COPY requirements-eval.txt .
RUN pip install --no-cache-dir -r requirements-eval.txt

COPY . .

# 기본 동작: 테스트 + Mock 종단 검증 (기준선 Balanced Accuracy 0.5000 이 나와야 정상)
CMD ["sh", "-c", "pytest tests/ -q && python eval/run_eval.py --backend mock:always_unknown --set smoke"]
