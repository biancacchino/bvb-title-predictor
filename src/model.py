import pandas as pd
from sklearn.linear_model import PoissonRegressor

# scikit-learn's default alpha=1.0 shrinks the coefficients so hard they barely move,
# a tiny alpha keeps the fit close to plain Poisson regression
REGULARIZATION = 1e-4

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
    return home_model, away_model

# Simplification: home and away goals are drawn independently, which predicts slightly too few draws
# (22% vs 25% real in 2023-2025). Dixon-Coles adds a draw correction if the backtest needs it.
def predict_goals(models, matches):
    # expected goals for each match, these are the Poisson averages the simulation draws from
    home_model, away_model = models
    features = model_features(matches)
    return home_model.predict(features), away_model.predict(features)
