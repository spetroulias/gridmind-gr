"""Calendar-only load projections without recursively predicted lag inputs.

The final year is held out chronologically to estimate historical errors. The
error band is descriptive: it is not a coverage guarantee under future changes
in weather, demand, generation behind the meter or the economy.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

MAX_HORIZON_YEARS = 2
MIN_COVERAGE = 0.8


def calendar_features(timestamps):
    timestamps = pd.DatetimeIndex(timestamps)
    phase = 2 * np.pi * (timestamps.dayofyear - 1) / np.where(timestamps.is_leap_year, 366, 365)
    return pd.DataFrame({
        "hour": timestamps.hour,
        "weekday": timestamps.dayofweek,
        "month": timestamps.month,
        "annual_sin": np.sin(phase),
        "annual_cos": np.cos(phase),
    })


def build_model():
    return HistGradientBoostingRegressor(
        max_iter=150, max_leaf_nodes=20, l2_regularization=1.0,
        early_stopping=False, random_state=42,
    )


def prepare_history(historical):
    history = historical[["timestamp", "load_mwh"]].copy()
    history["timestamp"] = pd.to_datetime(history["timestamp"], errors="raise")
    if history.empty or history["timestamp"].isna().any():
        raise ValueError("Long-term forecasting requires historical load observations.")
    if history["timestamp"].duplicated().any():
        raise ValueError("Duplicate timestamps in long-term forecast history.")
    history = history.sort_values("timestamp").reset_index(drop=True)
    history["load_mwh"] = pd.to_numeric(history["load_mwh"], errors="coerce")
    history.loc[~np.isfinite(history["load_mwh"]) | (history["load_mwh"] < 500), "load_mwh"] = np.nan
    return history


def split_history(history):
    # One full seasonal cycle for validation, at least one for training.
    end = history["timestamp"].max().normalize() + pd.Timedelta(days=1)
    cutoff = end - pd.DateOffset(years=1)
    minimum_start = cutoff - pd.DateOffset(years=1)
    if history["timestamp"].min() > minimum_start:
        raise ValueError("Long-term forecasts need at least two years of load history, including a full held-out year.")
    valid = history.dropna(subset=["load_mwh"])
    train = valid[valid["timestamp"] < cutoff].copy()
    validation = valid[valid["timestamp"] >= cutoff].copy()
    # Check coverage month by month in the two most recent annual cycles.
    for left, right in [(minimum_start, cutoff), (cutoff, end)]:
        expected = pd.Series(1, index=pd.date_range(left, right, freq="h", inclusive="left")).groupby(lambda t: t.to_period("M")).sum()
        observed = valid[(valid.timestamp >= left) & (valid.timestamp < right)]
        counts = observed.groupby(observed.timestamp.dt.to_period("M")).size()
        if (counts.reindex(expected.index, fill_value=0) / expected < MIN_COVERAGE).any():
            raise ValueError("Long-term forecasts require at least 80% valid hourly coverage in each month of the latest two years.")
    return train, validation


def metrics(actual, predicted):
    errors = np.asarray(actual) - np.asarray(predicted)
    return {
        "mae_mwh": float(np.abs(errors).mean()),
        "rmse_mwh": float(np.sqrt(np.mean(errors ** 2))),
        "wape_pct": float(np.abs(errors).sum() / np.abs(actual).sum() * 100),
    }


def make_long_term_forecast(historical, target_date):
    history = prepare_history(historical)
    latest = history["timestamp"].max().normalize()
    target = pd.Timestamp(target_date).normalize()
    if target <= latest:
        raise ValueError("Choose a forecast date after the latest load observations.")
    if target > latest + pd.DateOffset(years=MAX_HORIZON_YEARS):
        raise ValueError(f"Forecast dates must be within {MAX_HORIZON_YEARS} years after the latest load observations.")
    training, validation = split_history(history)
    model = build_model()
    model.fit(calendar_features(training.timestamp), training.load_mwh.to_numpy())
    predicted = np.maximum(model.predict(calendar_features(validation.timestamp)), 0)
    residuals = np.abs(validation.load_mwh.to_numpy() - predicted)
    error_width = float(np.quantile(residuals, 0.90))
    validation_metrics = metrics(validation.load_mwh.to_numpy(), predicted)

    # Baseline: median historical load for the same month, weekday and hour.
    train_keys = calendar_features(training.timestamp)[["month", "weekday", "hour"]]
    train_keys["value"] = training.load_mwh.to_numpy()
    profile = train_keys.groupby(["month", "weekday", "hour"]).value.median()
    validation_keys = calendar_features(validation.timestamp)[["month", "weekday", "hour"]]
    baseline = profile.reindex(pd.MultiIndex.from_frame(validation_keys)).fillna(training.load_mwh.median()).to_numpy()

    final_history = history.dropna(subset=["load_mwh"])
    final_model = build_model()
    final_model.fit(calendar_features(final_history.timestamp), final_history.load_mwh.to_numpy())
    times = pd.date_range(target, periods=24, freq="h")
    forecast = np.maximum(final_model.predict(calendar_features(times)), 0)
    frame = pd.DataFrame({
        "timestamp": times,
        "forecast_mwh": forecast,
        "lower_mwh": np.maximum(forecast - error_width, 0),
        "upper_mwh": forecast + error_width,
        "forecast_type": "long_term_seasonal",
    })
    metadata = {
        "method": "Calendar-only gradient boosting",
        "trained_through": str(history.timestamp.max()),
        "horizon_days": (target - latest).days,
        "validation_start": str(validation.timestamp.min()),
        "validation_end": str(validation.timestamp.max()),
        "validation_count": len(validation),
        "validation_metrics": validation_metrics,
        "seasonal_baseline_metrics": metrics(validation.load_mwh.to_numpy(), baseline),
        "error_band_mwh": error_width,
        "error_band_description": "Point forecast plus/minus the 90th percentile of absolute errors on a held-out year; future coverage is not guaranteed.",
        "assumptions": "Historical calendar patterns persist. No future weather, holidays, economic growth or structural demand changes are modeled.",
    }
    return frame, metadata
