# Plan: finish the BVB title predictor

Claude implements from here on and explains each feature after it is built.
Every step ends with tests passing and a short explanation.
Commits, pushes and PRs wait for approval.

## Step 2g: last-5 form (`src/features.py`)
- [x] `add_form(matches)` builds the long table, takes the rolling mean of points over the last 5 finished matches, shifts it by one match, and fills forward per team.
- [x] Form carries over the summer (decided 2026-10-27).
- [x] A team's first-ever Bundesliga match has no form, so it gets the league average (about 1.4 points per match).
- [x] Adds `home_form_before` and `away_form_before` to `matches`.
- [x] Tests: form uses only earlier matches, unplayed matches get the latest form, and teams don't mix.
- Verify: no missing form values in 7038 rows, and BVB's 2020 numbers match a hand calculation.

## Step 3: match model (`src/model.py`)
- [x] Two Poisson regressions (scikit-learn `PoissonRegressor`), one for home goals and one for away goals.
- [x] Features: Elo difference, home form, away form.
- [x] Train on finished matches only.
- [x] `predict_goals(model, matches)` returns expected home and away goals per match.
- Verify: the average predicted goals are close to the real averages (about 1.6 home, 1.3 away), and the stronger team gets more expected goals.

## Step 4: Monte Carlo (`src/simulate.py`)
- [x] Draw goals from a Poisson distribution for every unplayed current-season match, 10,000 times (numpy, vectorized).
- [x] Build the table from finished results plus simulated results, with tiebreaks on points, goal difference, then goals scored.
- [x] Return the title probability for each team.
- [x] Ratings stay frozen inside a simulated season (simplification, noted in code).
- Verify: probabilities add up to 1, and a finished season gives 100% to the real champion.

## Step 5: backtest (`src/backtest.py`)
- [ ] For each past season, predict the title from the table at matchday 17, training only on earlier matches.
- [ ] Score with the Brier score against the real champion.
- [ ] Small grid search over `K_FACTOR`, `HOME_ADVANTAGE`, `SUMMER_KEEP`, `SECOND_DIVISION_RATING`.
- Verify: the model beats a baseline that gives the title to the current leader.

## Step 6: `main.py`, chart, README
- [ ] `python main.py` loads data, trains, simulates, and prints BVB's title chance plus the top 5.
- [ ] Bar chart of title chances saved to `docs/`.
- [ ] README: how it works, how to run it, results.

## Housekeeping
- [ ] Clear notebook outputs before committing (they add about 47k lines to the diff).
- [x] Remove the dead `match_1_to_34` line in `src/data.py` (done in the 2004-2007 score fix, PR #3).
- [ ] Update the ML vault after each step.
