import itertools
import numpy as np
import pandas as pd
from sklearn.metrics import mean_poisson_deviance
from src.features import add_elo, add_form
from src.model import train, predict_goals
from src.simulate import simulate_season

# 2004-2007 only warm up the Elo ratings, so predictions start in 2008
BACKTEST_SEASONS = range(2008, 2026)
CUTOFF_MATCHDAYS = [0, 9, 17, 25]

def brier_score(title_probability, champion):
    # squared error summed over all teams: 0 is a sure right answer, 2 is a sure wrong one
    outcome = (title_probability.index == champion).astype(float)
    return ((title_probability - outcome) ** 2).sum()

def hide_results_after(matches, season, matchday):
    # What we knew on that matchday: later seasons dropped, later results of this season unplayed.
    # The goals are wiped too, so no step can read a hidden result by accident.
    known = matches[matches.season <= season].copy()
    hidden = (known.season == season) & (known.matchday > matchday)
    known.loc[hidden, ['home_goals', 'away_goals']] = np.nan
    known.loc[hidden, 'finished'] = False
    return known

def table_leader(season_matches):
    # the real table needs no model: with only finished matches, one "simulation" is the table itself
    finished = season_matches[season_matches.finished]
    return simulate_season(finished, models=None, number_of_simulations=1).index[0]

def title_backtest(matches, seasons=BACKTEST_SEASONS, cutoffs=CUTOFF_MATCHDAYS, number_of_simulations=2000, elo_settings=None):
    # For every past season and cutoff, predict the title using only what was known then,
    # and compare with a baseline that gives 100% to the leader (or last champion at matchday 0).
    rows = []
    for season in seasons:
        champion = table_leader(matches[matches.season == season])
        previous_champion = table_leader(matches[matches.season == season - 1])
        for cutoff in cutoffs:
            known = add_form(add_elo(hide_results_after(matches, season, cutoff), **(elo_settings or {})))
            this_season = known[known.season == season]
            title_probability = simulate_season(this_season, train(known), number_of_simulations, random_seed=season)
            baseline_pick = table_leader(this_season) if cutoff > 0 else previous_champion
            rows.append({
                'season': season,
                'cutoff_matchday': cutoff,
                'champion': champion,
                'champion_probability': title_probability.get(champion, 0),
                'favourite': title_probability.index[0],
                'brier': brier_score(title_probability, champion),
                'baseline_pick': baseline_pick,
                'baseline_brier': 0 if baseline_pick == champion else 2,
            })
    return pd.DataFrame(rows)

def match_deviance(matches, seasons=BACKTEST_SEASONS, **elo_settings):
    # How well the model predicts goals in seasons it never trained on (lower is better).
    # Used for tuning: thousands of matches are a much steadier target than 18 champions.
    featured = add_form(add_elo(matches, **elo_settings))
    season_deviances = []
    for season in seasons:
        testing = featured[(featured.season == season) & featured.finished]
        expected_home_goals, expected_away_goals = predict_goals(train(featured[featured.season < season]), testing)
        season_deviances.append(
            mean_poisson_deviance(testing.home_goals, expected_home_goals)
            + mean_poisson_deviance(testing.away_goals, expected_away_goals)
        )
    return np.mean(season_deviances)

def tune(matches, grid, seasons=BACKTEST_SEASONS):
    # try every combination in grid, e.g. {'k_factor': [15, 20, 30]}, best first
    names = list(grid)
    rows = []
    for values in itertools.product(*grid.values()):
        settings = dict(zip(names, values))
        rows.append({**settings, 'deviance': match_deviance(matches, seasons, **settings)})
    return pd.DataFrame(rows).sort_values('deviance').reset_index(drop=True)
