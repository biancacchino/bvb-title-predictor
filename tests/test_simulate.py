import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.simulate import sample_scores, simulate_season

class ConstantModel:
    # stands in for a trained model: every team scores the same number of goals on average
    def __init__(self, average_goals):
        self.average_goals = average_goals
    def predict(self, features):
        return np.full(len(features), self.average_goals)

EQUAL_TEAMS = (ConstantModel(1.4), ConstantModel(1.4), 1.0)

def make_season(rows):
    # rows are (home_team, away_team, home_goals, away_goals), None goals = unplayed
    matches = pd.DataFrame(rows, columns=['home_team', 'away_team', 'home_goals', 'away_goals'])
    matches['finished'] = matches.home_goals.notna()
    matches['home_elo_before'] = 1500
    matches['away_elo_before'] = 1500
    matches['home_form_before'] = 1.4
    matches['away_form_before'] = 1.4
    return matches

def test_finished_season_gives_the_champion_100_percent():
    season = make_season([('A', 'B', 2, 0), ('B', 'C', 1, 1), ('C', 'A', 0, 1)])
    title_probability = simulate_season(season, EQUAL_TEAMS, number_of_simulations=100)
    assert title_probability['A'] == 1
    assert title_probability['B'] == 0

def test_goal_difference_breaks_a_tie_on_points():
    # A and B both have 5 points, B has the better goal difference.
    # B on purpose: a full tie goes to A alphabetically, so A winning would prove nothing.
    season = make_season([('A', 'C', 1, 0), ('B', 'C', 5, 0), ('A', 'B', 0, 0), ('C', 'A', 1, 1), ('C', 'B', 1, 1)])
    title_probability = simulate_season(season, EQUAL_TEAMS, number_of_simulations=100)
    assert title_probability['B'] == 1

def test_goals_scored_breaks_a_tie_on_goal_difference():
    # both have 5 points and +2, B scored 4 goals and A only 3
    season = make_season([('A', 'C', 2, 0), ('B', 'C', 3, 1), ('A', 'B', 0, 0), ('C', 'A', 1, 1), ('C', 'B', 1, 1)])
    title_probability = simulate_season(season, EQUAL_TEAMS, number_of_simulations=100)
    assert title_probability['B'] == 1

def test_probabilities_add_up_to_one_and_cover_every_team():
    season = make_season([('A', 'B', 1, 0), ('B', 'C', None, None), ('C', 'A', None, None), ('D', 'A', None, None)])
    title_probability = simulate_season(season, EQUAL_TEAMS, number_of_simulations=2000, random_seed=0)
    assert set(title_probability.index) == {'A', 'B', 'C', 'D'}
    assert title_probability.sum() == pytest.approx(1)

def test_equal_teams_with_only_unplayed_matches_split_the_title():
    teams = ['A', 'B', 'C', 'D']
    season = make_season([(home, away, None, None) for home in teams for away in teams if home != away])
    title_probability = simulate_season(season, EQUAL_TEAMS, number_of_simulations=20000, random_seed=0)
    assert title_probability.to_numpy() == pytest.approx([0.25] * 4, abs=0.02)

def test_same_seed_gives_the_same_result():
    season = make_season([('A', 'B', None, None), ('B', 'A', None, None)])
    first = simulate_season(season, EQUAL_TEAMS, number_of_simulations=500, random_seed=7)
    second = simulate_season(season, EQUAL_TEAMS, number_of_simulations=500, random_seed=7)
    assert first.equals(second)

def test_draw_weight_increases_the_simulated_draw_rate():
    random = np.random.default_rng(0)
    home_goals, away_goals = sample_scores(np.array([1.4]), np.array([1.4]), 20000, random, draw_weight=1.5)
    assert (home_goals == away_goals).mean() == pytest.approx(0.34, abs=0.015)
