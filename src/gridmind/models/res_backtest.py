import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from gridmind.models.res_forecast import FEATURE_COLS, build_model_data as load_model_data


RF_PARAMS = {
    "n_estimators": 300,
    "max_depth": 20,
    "min_samples_leaf": 2,
    "random_state": 42,
    "n_jobs": -1,
}

BACKTEST_START = pd.Timestamp("2026-01-01")
BACKTEST_END = pd.Timestamp("2026-09-01")


def build_model() -> RandomForestRegressor:
    return RandomForestRegressor(**RF_PARAMS)


def evaluate(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    denominator = np.abs(y_true).sum()
    wape = (
        np.nan
        if denominator == 0
        else np.abs(y_true - y_pred).sum() / denominator * 100
    )

    return {
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2),
        "WAPE": round(wape, 2) if not np.isnan(wape) else np.nan,
    }


def run_backtest(df: pd.DataFrame) -> pd.DataFrame:
    """Run an expanding-window monthly backtest for 2026."""
    month_starts = pd.date_range(
        start=BACKTEST_START,
        end=BACKTEST_END,
        freq="MS",
        inclusive="left",
    )

    results = []

    for test_start in month_starts:
        test_end = test_start + pd.offsets.MonthBegin(1)

        train = df[df["timestamp"] < test_start].copy()
        test = df[
            (df["timestamp"] >= test_start)
            & (df["timestamp"] < test_end)
        ].copy()

        if train.empty:
            print(f"Skipping {test_start:%Y-%m}: no training data.")
            continue

        if test.empty:
            print(f"Skipping {test_start:%Y-%m}: no test data.")
            continue

        model = build_model()
        model.fit(train[FEATURE_COLS], train["target"])

        predictions = model.predict(test[FEATURE_COLS])
        metrics = evaluate(test["target"], predictions)

        result = {
            "month": test_start.strftime("%Y-%m"),
            "train_rows": len(train),
            "test_rows": len(test),
            **metrics,
        }
        results.append(result)

        print(
            f"{result['month']} "
            f"| MAE: {result['MAE']:.2f} "
            f"| RMSE: {result['RMSE']:.2f} "
            f"| WAPE: {result['WAPE']:.2f}%"
        )

    results_df = pd.DataFrame(results)

    if results_df.empty:
        raise ValueError("RES backtest produced no results.")

    return results_df


def print_summary(results: pd.DataFrame) -> None:
    best = results.loc[results["WAPE"].idxmin()]
    worst = results.loc[results["WAPE"].idxmax()]

    print("\nRES backtest summary")
    print(f"Average MAE:  {results['MAE'].mean():.2f} MWh")
    print(f"Average RMSE: {results['RMSE'].mean():.2f} MWh")
    print(f"Average WAPE: {results['WAPE'].mean():.2f}%")
    print(f"Best month:   {best['month']} ({best['WAPE']:.2f}% WAPE)")
    print(f"Worst month:  {worst['month']} ({worst['WAPE']:.2f}% WAPE)")


def main() -> None:
    print("Loading model data from PostgreSQL...")
    df = load_model_data()

    print("\nRunning monthly walk-forward backtest...")
    results = run_backtest(df)
    print_summary(results)


if __name__ == "__main__":
    main()
