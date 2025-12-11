# Data Leakage Issue & Fix Plan

## The Problem

The current model achieves **84% accuracy**, but this result is **invalid** due to **data leakage**.

### What is Data Leakage?

Data leakage occurs when your training data contains information that would not be available at prediction time. In this case, we're using **post-game statistics** to predict game outcomes.

### Current Features (Problematic)

| Feature | Problem |
|---------|---------|
| `FG_PCT_home` | Field goal % *from this game* |
| `FT_PCT_home` | Free throw % *from this game* |
| `FG3_PCT_home` | 3-point % *from this game* |
| `AST_home` | Assists *from this game* |
| `REB_home` | Rebounds *from this game* |
| (same for away team) | ... |

These features **already encode the outcome**. A team that shoots 55% FG almost certainly won — so the model is just learning that correlation, not actually predicting anything.

## The Solution

Replace post-game stats with **pre-game features** — information available *before* the game starts.

### Recommended Features

#### 1. Rolling Averages (Last N Games)
- `FG_PCT_home_last5` — Average FG% over home team's last 5 games
- `PTS_home_last10` — Average points over home team's last 10 games
- Same for all relevant stats (AST, REB, FT_PCT, FG3_PCT, etc.)

#### 2. Season-to-Date Averages
- `FG_PCT_home_season` — Home team's season average FG%
- `WIN_PCT_home_season` — Home team's win percentage this season

#### 3. Home/Away Specific Performance
- `WIN_PCT_home_at_home` — Team's win % when playing at home
- `WIN_PCT_away_on_road` — Team's win % when playing away

#### 4. Rest & Schedule Factors
- `REST_DAYS_home` — Days since home team's last game
- `REST_DAYS_away` — Days since away team's last game
- `BACK_TO_BACK_home` — Boolean: is this a back-to-back game?

#### 5. Head-to-Head History
- `H2H_home_wins_season` — Home team wins vs this opponent this season

## Implementation Steps

### Step 1: Sort Data Chronologically
Ensure games are sorted by date before computing rolling features.

### Step 2: Compute Rolling Stats Per Team
For each game, calculate each team's stats using **only prior games**:

```python
def compute_rolling_stats(df, team_id, game_date, window=5):
    # Get team's games BEFORE this date
    prior_games = df[
        ((df['HOME_TEAM_ID'] == team_id) | (df['VISITOR_TEAM_ID'] == team_id)) &
        (df['GAME_DATE_EST'] < game_date)
    ].tail(window)
    # Compute averages...
```

### Step 3: Handle Early-Season Games
Games at the start of a season won't have enough history. Options:
- Drop first N games of each season
- Use season-to-date averages with fallback to league averages
- Use prior season stats as baseline

### Step 4: Retrain Model
After fixing features, expect accuracy to drop to **55-65%** range. This is realistic for NBA game prediction.

## Expected Results After Fix

| Metric | Current (Leaked) | Expected (Fixed) |
|--------|------------------|------------------|
| Accuracy | 84% | 55-65% |
| Baseline | 55.6% | 55.6% |
| Real Improvement | N/A | 0-10% |

A 60% accuracy would be a legitimate ~5% improvement over baseline — still valuable for sports prediction!

## Files to Modify

1. **02_data_preprocessing.ipynb** — Add rolling feature computation
2. **03_model_training.ipynb** — Update to use new features
3. **data/processed/** — Save new preprocessed dataset with rolling features
