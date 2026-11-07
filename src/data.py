from pathlib import Path
import pandas as pd
import requests

DATA_DIR = Path(__file__).resolve().parent.parent / 'data'
CURRENT_SEASON = 2026

def full_time_score(match):
    for result in match['matchResults']:
        # match by name: the type ID is 0 for every result up to 2007 and 2 for full time after
        if result['resultName'] == 'Endergebnis':
            return result['pointsTeam1'], result['pointsTeam2']
    return None, None

def fetch_season(year):
    response = requests.get(f'https://api.openligadb.de/getmatchdata/bl1/{year}', timeout=10)
    response.raise_for_status()
    matches = response.json()
    rows = []
    for match in matches:
        home_goals, away_goals = full_time_score(match)
        rows.append({
            'season': match['leagueSeason'],
            'matchday': match['group']['groupOrderID'],
            'date': match['matchDateTime'],
            'home_team': match['team1']['teamName'],
            'away_team': match['team2']['teamName'],
            'home_goals': home_goals,
            'away_goals': away_goals,
            'finished': match['matchIsFinished']
        })
    df = pd.DataFrame(rows)
    # drop the 2008 relegation playoff, only matchdays 1-34 are league matches
    return df[df['matchday'] <= 34]


def load_season(year):
    # if the cache exists and its not the current season read it
    path = DATA_DIR / f'{year}.csv'
    if path.exists() and year!=CURRENT_SEASON:
        return pd.read_csv(path)
    df = fetch_season(year)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df

# 2003 only has results for 20 of 306 matches in the API, so history starts in 2004
def load_all(start=2004, end=CURRENT_SEASON):
    # an empty list, append each load_season , then pd.concat and no yield
    dfs = []
    for year in range(start, end+1):
        print(f'Loading {year}')
        dfs.append(load_season(year))
    matches = pd.concat(dfs, ignore_index=True)
    matches['date'] = pd.to_datetime(matches['date'])
    # a finished match without a score breaks every feature after it, so stop here
    missing_scores = matches[matches.finished & matches.home_goals.isna()]
    if len(missing_scores) > 0:
        raise ValueError(f'{len(missing_scores)} finished matches have no score, seasons {sorted(missing_scores.season.unique().tolist())}')
    return matches