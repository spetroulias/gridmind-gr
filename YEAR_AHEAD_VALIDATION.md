# One-day year-ahead load projection

Implemented for the request to forecast a **specific day a year later**. It is available through chat and `/api/v1/forecasts/load?date=YYYY-MM-DD`. It does not change the load forecasting CLI's existing recursive behavior.

Requests beyond 14 days use `gridmind.models.long_term_load`; requests up to 14 days retain the existing model. The two-calendar-year horizon is an application limit, not an empirically established accuracy cutoff.

## Model and evaluation

- Model: histogram gradient boosting, 150 iterations, 20 leaf nodes, deterministic seed, no random early-stopping split.
- Inputs: hour, weekday, month, annual sine/cosine. All are available for the target date; no future actual load or predicted lags are used.
- Validation: final calendar year withheld in chronological order. The model is frozen for the whole validation year, then a separate model is trained on all valid history for the requested target.
- Baseline: training-history median load for the same month, weekday and hour.
- Cleaning: inherited load cutoff of 500 MWh; nonfinite values excluded. At least two years and 80% valid hourly coverage per month across the latest two annual cycles are required.

This uses calendar feature engineering of the kind described in the [official scikit-learn example](https://scikit-learn.org/stable/auto_examples/applications/plot_cyclical_feature_engineering.html). The specific electricity model and validation choices are project implementation decisions, not results established by that example.

## Real local-data check

Historical data: 2023-01-01 through 2026-08-31.

Validation: 2025-09-01 through 2026-08-31, 8,751 valid hourly observations. Model training for validation used only earlier observations.

| Measure | Calendar model | Seasonal median baseline |
|---|---:|---:|
| MAE (MWh per hourly interval) | 509.26 | 517.65 |
| RMSE (MWh per hourly interval) | 706.39 | 714.97 |
| WAPE | 11.30% | 11.49% |

A real request for **2027-09-12** (377 days after the latest data) generated all 24 target-hour estimates successfully. No database records were written.

## Error bounds and interpretation

The displayed band is the final-model point estimate plus/minus **1,120.81 MWh**, the 90th percentile of absolute validation errors, with the lower line clipped at zero. This width will be recomputed if the historical data change.

This is a descriptive error range, **not a calibrated guarantee of future coverage**. It pools hours and seasons from one held-out year. It does not grow with the requested horizon and does not capture future weather, holidays, economic shifts, demand growth or changing solar self-consumption. Do not sum bounds to obtain aggregate confidence intervals.

The model only narrowly beat the seasonal baseline in this one evaluation. More forecast origins, time periods and external predictors are needed before claiming dependable long-term accuracy. The held-out year does not validate the second year of the application's allowed horizon.

The repository's existing nominal 24-period/day convention remains in use. Daylight-saving interpretation still needs correction before exact clock-time accounting is supported.

## Automated checks

36 tests pass in the complete suite, including new tests for:

- future-only calendar inputs;
- chronological training/validation separation;
- exactly 24 target-hour estimates and ordered nonnegative bounds;
- insufficient and missing seasonal history;
- historical and beyond-limit target rejection;
- routing year-ahead requests through the long-term model in chat/API;
- the `Forecast load in one year` date expression.
