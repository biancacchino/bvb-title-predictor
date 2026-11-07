import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.backtest import brier_score, hide_results_after, table_leader

def test_brier_score_is_0_for_a_sure_right_answer_and_2_for_a_sure_wrong_one():
    assert brier_score(pd.Series({'A': 1.0, 'B': 0.0}), 'A') == 0
    assert brier_score(pd.Series({'A': 1.0, 'B': 0.0}), 'B') == 2

def test_brier_score_rewards_putting_more_on_the_champion():
    unsure = pd.Series({'A': 0.5, 'B': 0.5})
    leaning_right = pd.Series({'A': 0.8, 'B': 0.2})
    assert brier_score(leaning_right, 'A') < brier_score(unsure, 'A')

def make_matches():
    return pd.DataFrame({
        'season': [2010, 2011, 2011, 2012],
        'matchday': [34, 1, 2, 1],
        'home_team': ['A', 'A', 'B', 'A'],
        'away_team': ['B', 'B', 'A', 'B'],
        'home_goals': [1.0, 2.0, 3.0, 0.0],
        'away_goals': [0.0, 0.0, 1.0, 0.0],
        'finished': [True, True, True, True],
    })

def test_hide_results_after_keeps_only_what_was_known_at_the_cutoff():
    hidden = hide_results_after(make_matches(), season=2011, matchday=1)
    # later seasons are gone, earlier ones are untouched
    assert list(hidden.season) == [2010, 2011, 2011]
    assert list(hidden.finished) == [True, True, False]
    # the hidden result is really gone, not just flagged, so nothing can read it by accident
    assert np.isnan(hidden.home_goals.iloc[2]) and np.isnan(hidden.away_goals.iloc[2])

def test_hide_results_after_does_not_change_the_input():
    matches = make_matches()
    hide_results_after(matches, season=2011, matchday=0)
    assert matches.finished.all()

def test_table_leader_uses_points_then_goal_difference():
    season = pd.DataFrame({
        'home_team': ['A', 'B', 'C'],
        'away_team': ['C', 'C', 'A'],
        'home_goals': [1.0, 4.0, 0.0],
        'away_goals': [0.0, 0.0, 0.0],
        'finished': [True, True, True],
    })
    # A has 4 points, B has 3 points and the better goal difference: points come first
    assert table_leader(season) == 'A'
    # without the draw both have 3 points, so B's goal difference decides.
    # B on purpose: a broken tiebreak would fall to A alphabetically.
    assert table_leader(season.head(2)) == 'B'

def test_table_leader_ignores_unplayed_matches():
    season = pd.DataFrame({
        'home_team': ['A', 'B'], 'away_team': ['C', 'C'],
        'home_goals': [1.0, np.nan], 'away_goals': [0.0, np.nan], 'finished': [True, False],
    })
    assert table_leader(season) == 'A'
