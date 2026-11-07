import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import skellam
from sklearn.linear_model import PoissonRegressor

# scikit-learn's default alpha=1.0 shrinks the coefficients so hard they barely move,
# a tiny alpha keeps the fit close to plain Poisson regression
REGULARIZATION = 1e-4

def draw_probability(expected_home_goals, expected_away_goals):
    # the goal difference of two independent Poisson scores follows a Skellam distribution,
    # so a draw is a goal difference of 0
    return skellam.pmf(0, expected_home_goals, expected_away_goals)

def fit_draw_weight(expected_home_goals, expected_away_goals, home_goals, away_goals):
    # Reweighting a draw by w changes its probability from p to wp / (1 - p + wp).
    # Find the one global weight whose average prediction matches the training draw rate.
    base_probability = draw_probability(expected_home_goals, expected_away_goals)
    observed_rate = np.mean(home_goals == away_goals)

    def rate_gap(weight):
        return np.mean(weight * base_probability / (1 - base_probability + weight * base_probability)) - observed_rate
    # brentq raises if no weight in the range works, instead of returning a bad one
    return brentq(rate_gap, 0.01, 100.0)

def model_features(matches):
    return pd.DataFrame({
        # per 100 so the coefficient is easy to read: how much log-goals change per 100 Elo points
        'elo_difference': (matches.home_elo_before - matches.away_elo_before) / 100,
        'home_form': matches.home_form_before,
        'away_form': matches.away_form_before,
    })

def train(matches):
    # One Poisson model for home goals and one for away goals.
    # The home model's intercept learns home advantage on its own.
    finished = matches[matches.finished]
    features = model_features(finished)
    home_model = PoissonRegressor(alpha=REGULARIZATION).fit(features, finished.home_goals)
    away_model = PoissonRegressor(alpha=REGULARIZATION).fit(features, finished.away_goals)
    draw_weight = fit_draw_weight(home_model.predict(features), away_model.predict(features),
                                  finished.home_goals, finished.away_goals)
    return home_model, away_model, draw_weight

def predict_goals(models, matches):
    # expected goals for each match, these are the Poisson averages the simulation draws from
    home_model, away_model, _ = models
    features = model_features(matches)
    return home_model.predict(features), away_model.predict(features)
