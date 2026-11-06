import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.data import full_time_score

def test_full_time_score_reads_old_seasons():
    # up to 2007 the API marks every result as type 0
    match = {'matchResults': [
        {'resultTypeID': 0, 'resultName': 'Endergebnis', 'pointsTeam1': 3, 'pointsTeam2': 1},
        {'resultTypeID': 0, 'resultName': 'Halbzeit', 'pointsTeam1': 3, 'pointsTeam2': 0},
    ]}
    assert full_time_score(match) == (3, 1)

def test_full_time_score_reads_new_seasons():
    # from 2008 on the halftime result can come first
    match = {'matchResults': [
        {'resultTypeID': 1, 'resultName': 'Halbzeit', 'pointsTeam1': 0, 'pointsTeam2': 0},
        {'resultTypeID': 2, 'resultName': 'Endergebnis', 'pointsTeam1': 2, 'pointsTeam2': 1},
    ]}
    assert full_time_score(match) == (2, 1)

def test_full_time_score_is_none_before_kickoff():
    assert full_time_score({'matchResults': []}) == (None, None)
