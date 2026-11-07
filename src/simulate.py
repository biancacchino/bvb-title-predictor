import numpy as np
import pandas as pd
from src.features import match_points
from src.model import predict_goals

NUMBER_OF_SIMULATIONS = 10000

def simulate_season(season_matches, models, number_of_simulations=NUMBER_OF_SIMULATIONS, random_seed=None):
    # Play every unplayed match of one season many times and return each team's title probability.
    # Simplification: ratings stay frozen inside a simulated season, so a team on a simulated
    # winning run does not get stronger. This makes the outcomes a little less spread out than reality.
    teams = np.array(sorted(set(season_matches.home_team) | set(season_matches.away_team)))
    unplayed = ~season_matches.finished.to_numpy()

    # one row per simulation, one column per match: real goals for finished matches,
    # random Poisson goals for unplayed ones
    home_goals = np.tile(season_matches.home_goals.to_numpy(dtype=float), (number_of_simulations, 1))
    away_goals = np.tile(season_matches.away_goals.to_numpy(dtype=float), (number_of_simulations, 1))
    if unplayed.any():
        expected_home_goals, expected_away_goals = predict_goals(models, season_matches[unplayed])
        random = np.random.default_rng(random_seed)
        home_goals[:, unplayed] = random.poisson(expected_home_goals, (number_of_simulations, unplayed.sum()))
        away_goals[:, unplayed] = random.poisson(expected_away_goals, (number_of_simulations, unplayed.sum()))

    # 1 where a team is the home (or away) side of a match, so a matrix product adds
    # every match's points and goals to the right team in all simulations at once
    home_side = season_matches.home_team.to_numpy()[:, None] == teams
    away_side = season_matches.away_team.to_numpy()[:, None] == teams

    points = match_points(home_goals, away_goals) @ home_side + match_points(away_goals, home_goals) @ away_side
    goals_scored = home_goals @ home_side + away_goals @ away_side
    goals_conceded = away_goals @ home_side + home_goals @ away_side

    # Bundesliga tiebreaks: points, then goal difference, then goals scored.
    # Each goal difference and goals total is far below 1000, so one number sorts all three at once.
    # Simplification: the later tiebreaks (head-to-head) are skipped, a full tie goes to the first team alphabetically.
    goal_difference = goals_scored - goals_conceded
    ranking_score = points * 1_000_000 + (goal_difference + 500) * 1000 + goals_scored
    champions = ranking_score.argmax(axis=1)

    title_count = np.bincount(champions, minlength=len(teams))
    title_probability = pd.Series(title_count / number_of_simulations, index=teams)
    return title_probability.sort_values(ascending=False)
