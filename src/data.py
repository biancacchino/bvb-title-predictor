from pathlib import Path
import pandas as pd
import requests

DATA_DIR = Path(__file__).resolve().parent.parent / 'data'
CURRENT_SEASON = 2026

def full_time_score(match):
    for result in match['matchResults']:
        # 2 is the score after full time, 1 is halftime
        if result['resultTypeID'] == 2:
            return result['pointsTeam1'], result['pointsTeam2']
    return None, None   

def fetch_season(year):
    r   = requests.get(f'https://api.openligadb.de/getmatchdata/bl1/{year}', timeout=10)
    r.raise_for_status()
    matches = r.json()
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
        match_1_to_34 = match['group']['groupOrderID'] <= 34
    df = pd.DataFrame(rows)
    return df[df['matchday'] <= 34]


def load_season(year):
    # if the cache exists and its not the current season read it
    path = DATA_DIR / f'{year}.csv'
    if path.exists() and year!=CURRENT_SEASON:
        return pd.read_csv(path)
    else: # fetch -> mkdir -> save -> return
        df = fetch_season(year)
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(path,index=False)
        return df

def load_all(start=2003, end=CURRENT_SEASON):
    # an empty list, append each load_season , then pd.concat and no yield
    dfs = []
    for year in range(start, end+1):
        print(f'Loading {year}')
        dfs.append(load_season(year))
    return pd.concat(dfs)