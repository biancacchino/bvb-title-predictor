import pandas as pd

# Tuned in the backtest (src/backtest.py) on how well the model predicts goals in unseen seasons.
HOME_ADVANTAGE = 60   # Elo points added to the home team, barely matters because the match model learns home advantage itself
K_FACTOR = 15         # how far one match can move a rating
LEAGUE_AVERAGE = 1500
SECOND_DIVISION_RATING = 1450
SUMMER_KEEP = 0.9     # share of a team's distance from the average it keeps over the summer
FORM_WINDOW = 5
FIRST_MATCH_FORM = 1.4   # about the league average points per match, for a team with no history

def expected_score(home_rating, away_rating, home_advantage=HOME_ADVANTAGE):
    return 1 / (1 + 10 ** ((away_rating - home_rating - home_advantage) / 400))

def actual_score(home_goals, away_goals):
    # 1 for a home win, 0.5 for a draw, 0 for an away win
    if home_goals > away_goals:
        return 1
    elif home_goals == away_goals:
        return 0.5
    else:
        return 0

def update(home_rating, away_rating, home_goals, away_goals, k_factor=K_FACTOR, home_advantage=HOME_ADVANTAGE):
    expected = expected_score(home_rating, away_rating, home_advantage)
    actual = actual_score(home_goals, away_goals)
    rating_change = k_factor * (actual - expected)
    return home_rating + rating_change, away_rating - rating_change

def add_elo(matches, k_factor=K_FACTOR, home_advantage=HOME_ADVANTAGE, summer_keep=SUMMER_KEEP,
            second_division_rating=SECOND_DIVISION_RATING):
    # Return a copy of matches with home_elo_before and away_elo_before columns.
    # The settings are arguments so the backtest can try other values.
    matches = matches.copy()
    matches = matches.sort_values('date').reset_index(drop=True)

    home_elo_before = []
    away_elo_before = []
    previous_season = None
    first_season = matches.season.min()
    ratings = {team: LEAGUE_AVERAGE for team in set(matches.loc[matches.season == first_season, 'home_team'])}
    for match in matches.itertuples():
        if match.season != previous_season:
            teams_this_season = set(matches.loc[matches.season == match.season, 'home_team'])
            for team, rating in ratings.items():
                target = LEAGUE_AVERAGE if team in teams_this_season else second_division_rating
                ratings[team] = target + summer_keep * (rating - target)
            previous_season = match.season

        home_rating = ratings.get(match.home_team, second_division_rating)
        away_rating = ratings.get(match.away_team, second_division_rating)

        # 1. save BEFORE updating (no leakage)
        home_elo_before.append(home_rating)
        away_elo_before.append(away_rating)

        # 2. update only if the match has been played
        if match.finished:
            new_home_rating, new_away_rating = update(
                home_rating, away_rating, match.home_goals, match.away_goals, k_factor, home_advantage
            )
            ratings[match.home_team] = new_home_rating
            ratings[match.away_team] = new_away_rating

    matches['home_elo_before'] = home_elo_before
    matches['away_elo_before'] = away_elo_before
    return matches


def match_points(goals_for, goals_against):
    return (goals_for > goals_against) * 3 + (goals_for == goals_against)

def add_form(matches):
    # Return a copy of matches with home_form_before and away_form_before:
    # average points over each team's last 5 finished matches, home and away together.
    matches = matches.sort_values('date').reset_index(drop=True)

    # one row per team per match, so a team's home and away matches line up in date order.
    # Unplayed matches get NaN points, so they don't count as losses.
    home_rows = pd.DataFrame({
        'match_index': matches.index,
        'side': 'home',
        'team': matches.home_team,
        'points': match_points(matches.home_goals, matches.away_goals).where(matches.finished),
    })
    away_rows = pd.DataFrame({
        'match_index': matches.index,
        'side': 'away',
        'team': matches.away_team,
        'points': match_points(matches.away_goals, matches.home_goals).where(matches.finished),
    })
    # matches is sorted by date, so match_index order is date order
    team_matches = pd.concat([home_rows, away_rows], ignore_index=True).sort_values('match_index', kind='stable')

    # form after each finished match, including that match
    finished_rows = team_matches[team_matches.points.notna()]
    team_matches['form_after'] = (
        finished_rows
        .groupby('team')
        .points
        .transform(lambda team_points: team_points.rolling(FORM_WINDOW, min_periods=1).mean())
    )

    # fill forward so unplayed matches get the latest known form,
    # then shift by one match so a match never sees its own result (no leakage)
    team_matches['form_before'] = (
        team_matches.groupby('team').form_after
        .transform(lambda form: form.ffill().shift())
        .fillna(FIRST_MATCH_FORM)
    )

    form_by_side = team_matches.pivot(index='match_index', columns='side', values='form_before')
    matches['home_form_before'] = form_by_side['home']
    matches['away_form_before'] = form_by_side['away']
    return matches
