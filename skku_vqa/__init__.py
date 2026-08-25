"""skku_vqa — 2026 성균관대학교 멀티모달 AI 챌린지 추론 파이프라인 (그린필드 v2).

설계 원칙 (대회 규칙을 1순위 제약으로):
  1. train 코드와 inference 코드를 물리적으로 분리(train/ vs inference/).
  2. 최종 답은 LLM이 생성한 텍스트에서 '추출'한다(parsing.parse_answer).
     unknown 옵션은 파싱 폴백·진단용일 뿐 정답 결정 수단이 아니다.
  3. 추론은 A6000에서 평균 0.5초/샘플 — vLLM 연속 배치 백엔드로 충족.
  4. 2026-06-01 이전 공개된 오픈 가중치만 사용(기본 Qwen3-VL-8B-Instruct, Apache-2.0, 2025-10).

이 패키지는 `pip install` 없이도 동작하도록 설계됐다. 진입 스크립트
(inference/·eval/·train/)가 pipeline 루트를 sys.path에 추가한 뒤
`from skku_vqa import ...` 로 import 한다.
"""
from __future__ import annotations

__version__ = "2.0.0"
