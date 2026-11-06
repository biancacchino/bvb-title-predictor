import sys
from pathlib import Path
import pandas as pd
import pytest
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.features import update, add_elo, add_form, FIRST_MATCH_FORM

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


def make_matches(rows):
    # rows are (date, home_team, away_team, home_goals, away_goals), None goals = unplayed
    matches = pd.DataFrame(rows, columns=['date', 'home_team', 'away_team', 'home_goals', 'away_goals'])
    matches['date'] = pd.to_datetime(matches['date'])
    matches['season'] = 2003
    matches['finished'] = matches.home_goals.notna()
    return matches

def test_add_form_only_uses_earlier_matches():
    matches = make_matches([
        ('2003-08-01', 'A', 'B', 2, 0),
        ('2003-08-08', 'A', 'B', 0, 1),
    ])
    result = add_form(matches)
    # nobody has played before the first match
    assert result['home_form_before'].iloc[0] == FIRST_MATCH_FORM
    # A lost the second match, but its form only knows about the 2-0 win
    assert result['home_form_before'].iloc[1] == 3
    assert result['away_form_before'].iloc[1] == 0

def test_add_form_counts_home_and_away_matches_for_the_same_team():
    matches = make_matches([
        ('2003-08-01', 'A', 'B', 1, 1),   # A draw at home
        ('2003-08-08', 'C', 'A', 0, 2),   # A win away
        ('2003-08-15', 'A', 'C', 0, 0),
    ])
    result = add_form(matches)
    assert result['home_form_before'].iloc[2] == pytest.approx((1 + 3) / 2)
    # C only lost to A, B's draw does not leak into C's form
    assert result['away_form_before'].iloc[2] == 0

def test_add_form_averages_only_the_last_5_matches():
    # A loses the first match, then wins 5, so the loss drops out of the window
    rows = [('2003-08-01', 'A', 'B', 0, 1)]
    rows += [(f'2003-08-{day:02d}', 'A', 'B', 1, 0) for day in range(2, 7)]
    rows += [('2003-08-07', 'A', 'B', 0, 0)]
    result = add_form(make_matches(rows))
    assert result['home_form_before'].iloc[6] == 3

def test_add_form_gives_unplayed_matches_the_latest_form():
    matches = make_matches([
        ('2003-08-01', 'A', 'B', 2, 0),
        ('2003-08-08', 'A', 'B', None, None),
        ('2003-08-15', 'B', 'A', None, None),
    ])
    result = add_form(matches)
    # both unplayed matches see the form after the 2-0, not a gap
    assert list(result['home_form_before'].iloc[1:]) == [3, 0]
    assert list(result['away_form_before'].iloc[1:]) == [0, 3]
