import sys
from pathlib import Path
import pandas as pd
import pytest
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.features import update, add_elo

def test_home_win_gains_less_than_half_k():
    new_home_rating, new_away_rating = update(1500, 1500, 2, 0)
    assert round(new_home_rating, 1) == 1508.3
    assert round(new_away_rating, 1) == 1491.7

def test_home_draw_loses_points():
    new_home_rating, new_away_rating = update(1500, 1500, 0, 0)
    assert round(new_home_rating, 1) == 1498.3
    assert round(new_away_rating, 1) == 1501.7

def test_update_is_zero_sum():
    new_home_rating, new_away_rating = update(1600, 1450, 1, 3)
    assert new_home_rating + new_away_rating == pytest.approx(3050.0)

def test_add_elo_saves_rating_before_the_match():
    # B also plays at home later so both teams are in the 2003 squad and start at 1500
    matches = pd.DataFrame({
        'season': [2003, 2003, 2003],
        'date': pd.to_datetime(['2003-08-01', '2003-08-08', '2003-08-15']),
        'home_team': ['A', 'A', 'B'],
        'away_team': ['B', 'B', 'A'],
        'home_goals': [2, 2, 0],
        'away_goals': [0, 0, 2],
        'finished': [True, True, True],
    })
    result = add_elo(matches)
    # first match: both start at the league average
    assert result['home_elo_before'].iloc[0] == 1500
    # second match: A gained 8.3 for the 2-0, B dropped the same amount
    assert round(result['home_elo_before'].iloc[1], 1) == 1508.3
    assert round(result['away_elo_before'].iloc[1], 1) == 1491.7

