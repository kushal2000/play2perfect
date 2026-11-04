"""A small number of finished episodes must not inflate a cohort score."""

from play2perfect_mjlab.evaluation_metrics import declared_start_metrics


def test_unfinished_and_early_episodes_remain_in_denominator():
    result = declared_start_metrics(
        goals=[2, 2, 1, 2],
        withdrawn=[True, True, False, True],
        valid=[True, False, False, True],
        total_goals=2,
        contact_free=[True, True, False, False],
    )
    assert result["num_starts"] == 4
    assert result["all_goals_hit_rate"] == 0.5
    assert result["withdrawal_rate"] == 0.5
    assert result["contact_free_withdrawal_count"] == 1
    assert result["contact_free_withdrawal_rate"] == 0.25
