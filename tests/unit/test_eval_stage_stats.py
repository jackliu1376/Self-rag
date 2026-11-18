from eval.runner import _stage_stats


def test_stage_stats_ignores_stages_that_did_not_run() -> None:
    assert _stage_stats([0.0, 0.0, 100.0, 200.0, 300.0]) == (200.0, 3)


def test_stage_stats_reports_zero_samples_for_unused_stage() -> None:
    assert _stage_stats([0.0, 0.0]) == (0.0, 0)
