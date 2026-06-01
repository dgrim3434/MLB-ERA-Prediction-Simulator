# Week 14 Capstone: ERA Prediction Simulator

## Overview

This notebook is the **Week 14 Capstone project**, with a primary focus on applying **linear regression** to a real baseball analytics problem. The goal of the project is to build a model that predicts a pitcher's **next-season ERA** using pitching metrics from the previous season.

All data used in this project was extracted from **FanGraphs**, including standard pitching statistics, Statcast-based metrics, and FanGraphs Plus metrics.

The notebook is structured as a full machine learning workflow: data loading, cleaning, feature selection, exploratory data analysis, multicollinearity reduction, model training, model tuning, evaluation, and model saving.

## Project Objective

The main objective is to answer the following question:

> Can a pitcher's current-season performance metrics be used to predict their ERA in the following season?

This is a difficult prediction problem because ERA is noisy and can be affected by defense, ballpark factors, bullpen support, sequencing, injuries, and luck. The notebook first checks the direct relationship between current-season ERA and next-season ERA, then builds regression models using a wider set of pitching indicators.

## Data Source

The data was extracted from **FanGraphs** for the 2018 through 2025 MLB seasons.

The notebook uses three groups of FanGraphs data:

- Standard pitcher statistics
- Statcast pitcher statistics
- FanGraphs Plus pitcher statistics

The expected file format is:

```text
standard_pitcher_stats_2018.csv
standard_pitcher_stats_2019.csv
...
standard_pitcher_stats_2025.csv

statcast_pitcher_stats_2018.csv
statcast_pitcher_stats_2019.csv
...
statcast_pitcher_stats_2025.csv

plus_pitcher_stats_2018.csv
plus_pitcher_stats_2019.csv
...
plus_pitcher_stats_2025.csv
```

Each CSV file is expected to be located in the same working directory as the notebook.

## Target Variable

The target variable is **next-season ERA**.

For each pitcher-season row, the notebook uses the pitcher's statistics from year `t` to predict their ERA in year `t + 1`. For example, a pitcher's 2023 statistics are used to predict their 2024 ERA.

This target setup makes the project more realistic than simply predicting same-season ERA, because the model is attempting to forecast future performance rather than explain already-observed results.

## Methodology

### 1. Data Loading

The notebook loads FanGraphs data from 2018 through 2025 and appends each season into combined DataFrames for standard, Statcast, and Plus statistics.

### 2. Duplicate Column Removal

Because the three FanGraphs datasets contain overlapping columns, duplicate columns are removed before merging. `PlayerId` and `year` are preserved so the datasets can be joined correctly.

### 3. Dataset Joining and Target Creation

The notebook merges the standard, Statcast, and Plus datasets on:

```text
PlayerId, year
```

Then it creates the prediction target by shifting ERA forward one season. Rows from 2025 are dropped as input rows because there is no following season available in the dataset to use as a target.

### 4. Exploratory Data Analysis

The notebook includes several EDA steps, including:

- ERA distribution
- Mean ERA by season
- Feature correlation heatmap
- Scatterplots between ERA and important pitching metrics
- Initial correlation between current ERA and next-season ERA

The direct correlation between current-season ERA and next-season ERA was approximately **0.20**, showing that next-season ERA is only weakly related to current ERA by itself.

### 5. Feature Removal

Several groups of features are removed before modeling:

#### Target leakage features

These columns are removed because they are directly tied to ERA or are too closely related to the target:

- `ERA-`
- `ER`
- `R`
- `xERA`

#### Opportunity, role, and luck-driven features

Some features are removed because they reflect team context, pitcher role, opportunity, or randomness more than underlying pitching skill:

- Wins and losses
- Saves, holds, blown saves
- Complete games and shutouts
- Batters faced
- BABIP+ and LOB%+

#### Redundant counting and rate statistics

Additional count-based and duplicated rate statistics are removed to reduce redundancy and avoid giving the model multiple versions of the same signal.

### 6. Multicollinearity Reduction

The notebook checks feature correlation and uses **Variance Inflation Factor (VIF)** to identify highly collinear features. Features with very high VIF scores are removed to make the linear regression model more stable and interpretable.

After feature selection, the final model uses the following candidate predictors:

```text
QS, IP, LA, HardHit%, K/BB+, HR/9+, K%+, xFIP-, LD%+
```

### 7. Train, Validation, and Test Split

The project uses a time-based split instead of a random split:

- Training set: seasons before 2023
- Validation set: 2023
- Test set: 2024

This is the correct structure for a forecasting problem because it prevents future seasons from leaking into earlier model training.

### 8. Preprocessing Pipeline

The modeling pipeline uses:

- Mean imputation for missing values
- Standard scaling for numeric features
- A regression model as the final estimator

Using a pipeline keeps preprocessing consistent across training, validation, testing, and future predictions.

## Models Used

Because this capstone focused on **linear regression**, the main models are linear regression variants:

- Linear Regression baseline
- Ridge Regression
- Lasso Regression
- ElasticNet Regression

The notebook also includes Random Forest and XGBoost sections as comparison models, but the main focus remains on linear modeling and regularization.

## Evaluation Metrics

The models are evaluated using:

- **MAE**: Mean Absolute Error
- **RMSE**: Root Mean Squared Error
- **R²**: Explained variance
- **MAPE**: Mean Absolute Percentage Error

MAE is especially useful here because it represents the average prediction error in ERA points.

## Validation Results

The best executed validation result came from **ElasticNet Regression**.

| Model | MAE | RMSE | R² | MAPE |
|---|---:|---:|---:|---:|
| Linear Regression | 0.898 | 1.160 | 0.054 | 0.263 |
| Ridge Regression | 0.897 | 1.159 | 0.055 | 0.263 |
| Lasso Regression | 0.890 | 1.157 | 0.059 | 0.262 |
| ElasticNet Regression | 0.889 | 1.156 | 0.061 | 0.262 |

The results show that regularization slightly improves performance compared to the baseline linear regression model. However, the low R² also shows that next-season ERA is difficult to predict using only the available previous-season pitching metrics.

## Best Model

Based on validation performance, the notebook selects **ElasticNet Regression** as the best model.

Best ElasticNet parameters:

```text
alpha = 0.1
l1_ratio = 0.5
```

The selected model is then evaluated on the test set and saved using `joblib`.

## Test Results

| Model | MAE | RMSE | R² | MAPE |
|---|---:|---:|---:|---:|
| Best Model: ElasticNet | 1.181 | 1.160 | 0.131 | 0.266 |


## Model Output

The notebook saves the final trained model as:

```text
best_model.joblib
```

This file can be loaded later and used to make ERA predictions on new pitcher data with the same feature structure.

## How to Run the Notebook

1. Place the notebook and all FanGraphs CSV files in the same directory.
2. Make sure the required Python libraries are installed.
3. Run the notebook from top to bottom.
4. Review the validation results and test-set results.
5. Use the saved `best_model.joblib` file for future predictions.

## Required Libraries

The notebook uses the following Python libraries:

```text
pandas
numpy
matplotlib
seaborn
scikit-learn
statsmodels
scipy
xgboost
joblib
```

## Key Takeaways

- Predicting next-season ERA is much harder than explaining same-season ERA.
- Current-season ERA alone has weak predictive power for next-season ERA.
- Regularized linear models performed slightly better than basic linear regression.
- ElasticNet was the best-performing linear model on the validation set.
- The low R² suggests that future improvements should include more context, such as innings thresholds, injury history, pitch-level data, park factors, team defense, and rolling multi-year performance trends.

## Future Improvements

The project could be improved by:

- Adding pitcher age and handedness
- Adding park-adjusted metrics
- Adding team defense indicators
- Including rolling multi-year averages
- Separating starters and relievers into different models
- Adding minimum innings filters
- Testing additional non-linear models more thoroughly
- Saving final test metrics directly in the notebook output
- Refactoring feature removal into a reusable configuration section

## Conclusion

This Week 14 Capstone demonstrates a complete linear regression workflow using real FanGraphs baseball data. The project shows how to build a forward-looking ERA prediction model, handle feature leakage, reduce multicollinearity, compare regularized regression models, and save the best model for future use.

While the final model has limited predictive power, that result is meaningful. It reflects the reality that pitcher ERA is noisy and difficult to forecast. The project still provides a strong foundation for future baseball analytics work using more advanced features and modeling approaches.
