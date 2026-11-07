from pathlib import Path
import matplotlib
matplotlib.use('Agg')   # save to a file, no window
import matplotlib.pyplot as plt
import pandas as pd
from src.data import load_all, CURRENT_SEASON
from src.features import add_elo, add_form
from src.model import train
from src.simulate import simulate_season

TEAM = 'Borussia Dortmund'
PROJECT_DIR = Path(__file__).resolve().parent
CHART_PATH = PROJECT_DIR / 'docs' / 'title_chances.png'
MINIMUM_CHANCE_IN_CHART = 0.005   # teams below 0.5% are folded into one bar

TEAM_COLOR = '#B08900'      # BVB gold, dark enough for 3:1 contrast on the background
OTHER_COLOR = '#8A9099'     # gray so the other teams stay in the background
INK = '#0b0b0b'
MUTED_INK = '#52514e'
BACKGROUND = '#fcfcfb'

def completed_matchdays(season_matches):
    finished_per_matchday = season_matches.groupby('matchday').finished.sum()
    return int((finished_per_matchday == 9).sum())

def save_chart(title_probability, matchday, path=CHART_PATH):
    # the contenders, plus everyone else folded into one bar so the chart stays readable
    top = title_probability[title_probability >= MINIMUM_CHANCE_IN_CHART]
    other = title_probability[title_probability < MINIMUM_CHANCE_IN_CHART]
    bars = pd.concat([top, pd.Series({f'{len(other)} other teams': other.sum()})])[::-1]

    figure, axis = plt.subplots(figsize=(8, 3.2), facecolor=BACKGROUND)
    axis.set_facecolor(BACKGROUND)
    colors = [TEAM_COLOR if team == TEAM else OTHER_COLOR for team in bars.index]
    axis.barh(bars.index, bars * 100, color=colors, height=0.6)
    for position, chance in enumerate(bars):
        axis.text(chance * 100 + 1, position, f'{chance:.1%}', va='center', fontsize=10, color=INK)

    axis.set_xlim(0, 100)
    axis.set_xlabel('Title chance (%)', color=MUTED_INK)
    axis.tick_params(colors=MUTED_INK, length=0)
    for label in axis.get_yticklabels():
        label.set_color(INK)
        label.set_fontweight('bold' if label.get_text() == TEAM else 'normal')
    for side in ['top', 'right', 'left']:
        axis.spines[side].set_visible(False)
    axis.spines['bottom'].set_color('#d8d7d2')
    axis.set_title(f'Bundesliga {CURRENT_SEASON}-{(CURRENT_SEASON + 1) % 100:02d} title chances after matchday {matchday}',
                   loc='left', color=INK, fontsize=12, fontweight='bold')

    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path, dpi=150, facecolor=BACKGROUND)
    plt.close(figure)

def main():
    matches = add_form(add_elo(load_all()))
    season_matches = matches[matches.season == CURRENT_SEASON]
    title_probability = simulate_season(season_matches, train(matches))
    matchday = completed_matchdays(season_matches)

    print(f'Bundesliga {CURRENT_SEASON}-{(CURRENT_SEASON + 1) % 100:02d} after matchday {matchday}'
          f' ({season_matches.finished.sum()} of {len(season_matches)} matches played)\n')
    print(f'{TEAM} title chance: {title_probability[TEAM]:.1%}\n')
    print('Top 5:')
    for team, chance in title_probability.head(5).items():
        print(f'  {team:<26}{chance:6.1%}')

    save_chart(title_probability, matchday)
    print(f'\nChart saved to {CHART_PATH.relative_to(PROJECT_DIR)}')

if __name__ == '__main__':
    main()
