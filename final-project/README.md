# NBA Game Outcome Prediction

A machine learning project that predicts NBA game outcomes (home team win/loss) using historical team performance data.

## Overview

This project analyzes ~26,500 NBA games from 2003-2022 to predict whether the home team will win. We use rolling averages of team statistics as features to avoid data leakage and compare multiple classification models.

**Key Results:**

- **Baseline (always predict home win):** 55.6% accuracy
- **Best Model (Random Forest):** 58.6% accuracy (+3% improvement)
- **Most Important Feature:** Field Goal Percentage (FG%)

1. **Create virtual environment:**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies:**

   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn jupyter
   ```

3. **Download data:**

   ```bash
   python download_data.py
   ```

4. **Run notebooks in order:**
   - `01_data_exploration.ipynb` → Explore raw data
   - `02_data_cleaning.ipynb` → Clean and preprocess
   - `03_feature_engineering.ipynb` → Create prediction features
   - `04_model_training_linear_regression.ipynb` → Train logistic regression
   - `05_model_training_random_forest.ipynb` → Compare all models

## Results Summary

| Model               | Train Acc | Test Acc | vs Baseline |
| ------------------- | --------- | -------- | ----------- |
| Random Forest       | 70.2%     | 58.6%    | +2.9%       |
| Logistic Regression | 61.4%     | 58.3%    | +2.6%       |
| Decision Tree       | 67.2%     | 55.6%    | +0.0%       |
