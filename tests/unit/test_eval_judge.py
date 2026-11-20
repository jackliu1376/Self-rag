from eval.runner import _judge_model


def test_judge_model_is_opt_in(monkeypatch) -> None:
    monkeypatch.delenv("EVAL_JUDGE_MODEL", raising=False)
    assert _judge_model() == ""
    monkeypatch.setenv("EVAL_JUDGE_MODEL", "  cn:glm-5.3  ")
    assert _judge_model() == "cn:glm-5.3"
