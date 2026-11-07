import sys
from pathlib import Path
import pandas as pd
sys.path.append(str(Path(__file__).resolve().parent.parent))

from main import save_chart, completed_matchdays

def test_save_chart_writes_a_png(tmp_path):
    title_probability = pd.Series({'FC Bayern München': 0.8, 'Borussia Dortmund': 0.19, 'A': 0.006, 'B': 0.004})
    chart_path = tmp_path / 'chart.png'
    save_chart(title_probability, matchday=8, path=chart_path)
    assert chart_path.read_bytes()[:4] == b'\x89PNG'

def test_completed_matchdays_counts_only_full_matchdays():
    season = pd.DataFrame({'matchday': [1] * 9 + [2] * 9, 'finished': [True] * 9 + [True] + [False] * 8})
    assert completed_matchdays(season) == 1
