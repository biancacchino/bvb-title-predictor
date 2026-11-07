import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.model import draw_probability, fit_draw_weight, train, predict_goals

def make_matches(number_of_matches, elo_effect, random_seed=0):
    # fake matches where goals come from a known formula, so we know what the model should find
    random = np.random.default_rng(random_seed)
    elo_difference = random.uniform(-300, 300, number_of_matches)
    matches = pd.DataFrame({
        'home_elo_before': 1500 + elo_difference / 2,
        'away_elo_before': 1500 - elo_difference / 2,
        'home_form_before': 1.4,
        'away_form_before': 1.4,
        'finished': True,
    })
    matches['home_goals'] = random.poisson(np.exp(np.log(1.6) + elo_effect * elo_difference / 100))
    matches['away_goals'] = random.poisson(np.exp(np.log(1.2) - elo_effect * elo_difference / 100))
    return matches

def test_predicts_average_goals_when_teams_are_equal():
    matches = make_matches(5000, elo_effect=0.2)
    equal_teams = matches.head(1).assign(home_elo_before=1500, away_elo_before=1500)
    expected_home_goals, expected_away_goals = predict_goals(train(matches), equal_teams)
    assert expected_home_goals[0] == pytest.approx(1.6, abs=0.1)
    assert expected_away_goals[0] == pytest.approx(1.2, abs=0.1)

def test_stronger_team_gets_more_goals():
    models = train(make_matches(5000, elo_effect=0.2))
    strong_home_team = pd.DataFrame({
        'home_elo_before': [1700], 'away_elo_before': [1400],
        'home_form_before': [1.4], 'away_form_before': [1.4],
    })
    expected_home_goals, expected_away_goals = predict_goals(models, strong_home_team)
    # 300 Elo points = 3 * 0.2 on the log scale, so about 1.6 * e^0.6 = 2.9 home goals
    assert expected_home_goals[0] == pytest.approx(1.6 * np.exp(0.6), rel=0.1)
    assert expected_away_goals[0] < 1.2

def test_train_skips_unplayed_matches():
    matches = make_matches(500, elo_effect=0.2)
    unplayed = matches.head(10).assign(finished=False, home_goals=np.nan, away_goals=np.nan)
    # would crash on NaN goals if unplayed matches were used for training
    train(pd.concat([matches, unplayed], ignore_index=True))

def test_draw_weight_matches_the_observed_draw_rate():
    expected_home_goals = np.array([1.5, 1.2, 1.8, 1.0])
    expected_away_goals = np.array([1.2, 1.1, 1.0, 1.3])
    home_goals = np.array([0, 1, 2, 0])
    away_goals = np.array([0, 1, 1, 0])  # three draws from four matches

    weight = fit_draw_weight(expected_home_goals, expected_away_goals, home_goals, away_goals)
    base = draw_probability(expected_home_goals, expected_away_goals)
    calibrated = np.mean(weight * base / (1 - base + weight * base))
    assert calibrated == pytest.approx(0.75)
