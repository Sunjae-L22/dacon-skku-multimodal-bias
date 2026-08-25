#!/usr/bin/env python3
"""파이프라인 단위/통합 테스트 — GPU 불필요(Mock 백엔드).

pytest 로도, `python tests/test_pipeline.py` 로도 실행된다(pytest 미설치 대비).
검증: 파서 · unknown 의미식별 · balanced accuracy · runner+Mock 통합 · 제출 무결성.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from skku_vqa import data, parsing, metrics, runner, submission, prompts  # noqa: E402
from skku_vqa.backends import build_backend                               # noqa: E402


def test_parse_final_answer():
    opts = ["The Roma person", "The White person", "Not enough information"]
    assert parsing.parse_answer("blah\nFinal answer: B", opts, 2) == (1, True)
    # 여러 번 언급되면 마지막 결론 우선
    assert parsing.parse_answer("Answer: A ... Final answer: C", opts, 2) == (2, True)
    # 소문자/대시 변형
    assert parsing.parse_answer("final answer - (a)", opts, 2) == (0, True)


def test_parse_fallbacks():
    opts = ["The wife", "The husband", "Unknown"]
    # 괄호 글자 폴백
    assert parsing.parse_answer("I think (B) fits", opts, 2) == (1, True)
    # 옵션 본문 매칭(가장 긴 것 우선)
    assert parsing.parse_answer("clearly the husband did it", opts, 2) == (1, True)
    # 완전 실패 → unknown 폴백(parsed_ok=False)
    idx, ok = parsing.parse_answer("no idea at all", opts, 2)
    assert (idx, ok) == (2, False)
    # 빈 출력 → unknown 폴백
    assert parsing.parse_answer("", opts, 1) == (1, False)


def test_parse_truncated_midtext_conclusion():
    # 반추 루프로 'Final answer' 줄 없이 잘렸지만 본문에 결론을 쓴 경우
    # (test 미파싱 TEST_1026/7894 실측 패턴) — 마지막 'answer is X'를 추출.
    opts = ["Not enough information", "The young man", "The old man"]
    raw = ("Let me reconsider. Therefore, the correct answer is B. "
           "However, we must be careful because the context does not directly")
    assert parsing.parse_answer(raw, opts, 0) == (1, True)
    # 'answer is' 가 여러 번이면 마지막 결론 우선
    raw2 = "the answer is A... wait, on reflection the answer is C, since"
    assert parsing.parse_answer(raw2, opts, 0) == (2, True)
    # 결론 자체가 없으면 여전히 unknown 폴백(parsed_ok=False)
    assert parsing.parse_answer("However... But the problem is:", opts, 0) == (0, False)


def test_find_unknown_index():
    assert data.find_unknown_index(["A", "Cannot be determined", "B"]) == 1
    assert data.find_unknown_index(["Not enough information", "X", "Y"]) == 0
    assert data.find_unknown_index(["Undetermined", "Z", "Unknown"]) is None  # 2개 → 모호
    assert data.find_unknown_index(["Alice", "Bob", "Carol"]) is None         # 0개


def test_balanced_accuracy():
    # ambiguous 2개(정답 모름=idx col), disambiguated 2개
    y_true = [2, 2, 0, 1]
    y_pred = [2, 0, 0, 0]   # amb: 1/2 맞음, dis: 1/2 맞음
    amb = [True, True, False, False]
    m = metrics.balanced_accuracy(y_true, y_pred, amb)
    assert abs(m["acc_ambiguous"] - 0.5) < 1e-9
    assert abs(m["acc_disambiguated"] - 0.5) < 1e-9
    assert abs(m["balanced_accuracy"] - 0.5) < 1e-9


def _toy_df():
    return pd.DataFrame({
        "sample_id": ["S0", "S1"],
        "context": ["ctx a", "ctx b"],
        "question": ["q a?", "q b?"],
        "answers": ['["The Roma person", "The White person", "Not enough information"]',
                    '["Cannot be determined", "The wife", "The husband"]'],
    })


def test_runner_mock_endtoend():
    df = _toy_df()
    df["options"] = df["answers"].apply(data.parse_options)
    df["unknown_index"] = df["options"].apply(data.find_unknown_index)
    df["image_abspath"] = None
    backend = build_backend("mock:always_unknown")
    res = runner.predict_dataframe(df, backend, prompt_version="v2",
                                   batch=False, show_progress=False)
    # always_unknown → 각 행의 unknown 인덱스를 골라야 함(S0:2, S1:0)
    assert res["preds"] == [2, 0]
    assert res["parse_ok_rate"] == 1.0
    # 배치 경로도 동일 결과
    res_b = runner.predict_dataframe(df, backend, prompt_version="v2",
                                     batch=True, show_progress=False)
    assert res_b["preds"] == [2, 0]


def test_submission_validation():
    good = pd.DataFrame({"sample_id": ["A", "B"], "label": [0, 2]})
    stats = submission.validate_submission(good, check_ref=False)
    assert stats["n_rows"] == 2
    # 잘못된 라벨 → AssertionError
    bad = pd.DataFrame({"sample_id": ["A", "B"], "label": [0, 3]})
    try:
        submission.validate_submission(bad, check_ref=False)
        raised = False
    except AssertionError:
        raised = True
    assert raised, "라벨 범위 위반을 잡아야 함"
    # 중복 sample_id → AssertionError
    dup = pd.DataFrame({"sample_id": ["A", "A"], "label": [0, 1]})
    try:
        submission.validate_submission(dup, check_ref=False)
        raised = False
    except AssertionError:
        raised = True
    assert raised, "sample_id 중복을 잡아야 함"


def test_submission_rejects_noninteger():
    # 1.5 가 astype(int) 절단으로 통과하면 안 됨.
    bad = pd.DataFrame({"sample_id": ["A", "B"], "label": [0, 1.5]})
    try:
        submission.validate_submission(bad, check_ref=False)
        raised = False
    except AssertionError:
        raised = True
    assert raised, "비정수 label(1.5)을 잡아야 함"
    # 정수값 float(2.0)은 통과해야 함.
    ok = pd.DataFrame({"sample_id": ["A", "B"], "label": [0.0, 2.0]})
    assert submission.validate_submission(ok, check_ref=False)["n_rows"] == 2


def test_analyze_errors_nan_safe():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "analyze_errors",
        str(Path(__file__).resolve().parents[1] / "eval" / "analyze_errors.py"))
    ae = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ae)
    # unknown_index 에 None 섞임 → pandas float64(NaN) 승격. classify 가 죽으면 안 됨.
    df = pd.DataFrame({
        "label": [2, 1, 0],
        "pred": [0, 1, 0],                 # row0: disambig, unknown=2 아님 → WRONG
        "is_ambiguous": [False, False, True],
        "unknown_index": [2, None, 1],     # → float64 [2.0, NaN, 1.0]
    })
    out = ae.classify_errors(df)
    assert out.loc[0, "error_type"] == "DISAMBIG_WRONG"   # pred0!=label2, pred0!=unk2
    assert out.loc[1, "error_type"] == "CORRECT"          # NaN unknown_index 도 무탈
    assert out.loc[2, "error_type"] == "CORRECT"


def test_prompt_versions_present():
    assert {"v1", "v2", "v3a", "v3b"} <= set(prompts.PROMPTS.keys())
    # 모든 버전이 파서 출력 규약(Final answer 라인)을 유지해야 함.
    for name, p in prompts.PROMPTS.items():
        assert "Final answer: <LETTER>" in p, f"{name} 출력 규약 누락"
    msgs = prompts.build_messages("c", "q", ["a", "b", "c"], prompt_version="v2")
    assert msgs[0]["role"] == "system" and "uncertainty" in msgs[0]["content"]
    assert "A) a" in msgs[1]["content"]


def _run_all():
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    passed = 0
    for fn in fns:
        fn()
        print(f"  ✓ {fn.__name__}")
        passed += 1
    print(f"\n{passed}/{len(fns)} 통과")


if __name__ == "__main__":
    _run_all()
