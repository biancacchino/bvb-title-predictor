HOME_ADVANTAGE = 60   # Elo points added to the home team
K_FACTOR = 20         # how far one match can move a rating
LEAGUE_AVERAGE = 1500
SECOND_DIVISION_RATING = 1400
SUMMER_KEEP = 0.8

def expected_score(home_rating, away_rating):
    return 1 / (1 + 10 ** ((away_rating - home_rating - HOME_ADVANTAGE) / 400))

def actual_score(home_goals, away_goals):
    # 1 for a home win, 0.5 for a draw, 0 for an away win
    if home_goals > away_goals:
        return 1
    elif home_goals == away_goals:
        return 0.5
    else:
        return 0

def update(home_rating, away_rating, home_goals, away_goals):
    expected = expected_score(home_rating, away_rating)
    actual = actual_score(home_goals, away_goals)
    rating_change = K_FACTOR * (actual - expected)
    return home_rating + rating_change, away_rating - rating_change

def add_elo(matches):
    # Return a copy of matches with home_elo_before and away_elo_before columns.
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
                target = LEAGUE_AVERAGE if team in teams_this_season else SECOND_DIVISION_RATING
                ratings[team] = target + SUMMER_KEEP * (rating - target)
            previous_season = match.season

        home_rating = ratings.get(match.home_team, SECOND_DIVISION_RATING)
        away_rating = ratings.get(match.away_team, SECOND_DIVISION_RATING)

        # 1. save BEFORE updating (no leakage)
        home_elo_before.append(home_rating)
        away_elo_before.append(away_rating)

        # 2. update only if the match has been played
        if match.finished:
            new_home_rating, new_away_rating = update(
                home_rating, away_rating, match.home_goals, match.away_goals
            )
            ratings[match.home_team] = new_home_rating
            ratings[match.away_team] = new_away_rating

    matches['home_elo_before'] = home_elo_before
    matches['away_elo_before'] = away_elo_before
    return matches