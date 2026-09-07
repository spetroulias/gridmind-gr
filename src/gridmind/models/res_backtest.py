from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
)

from gridmind.models.res_forecast import (
    FEATURE_COLS,
    build_model,
    build_model_data,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]

RESULTS_PATH = (
    PROJECT_ROOT
    / "data"
    / "forecasts"
    / "res_backtest_results.csv"
)

BACKTEST_START = pd.Timestamp("2026-01-01")
BACKTEST_END = pd.Timestamp("2026-09-01")


def evaluate(y_true, y_pred):
    y_true = np.asarray(
        y_true,
        dtype=float,
    )

    y_pred = np.asarray(
        y_pred,
        dtype=float,
    )

    mae = mean_absolute_error(
        y_true,
        y_pred,
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            y_pred,
        )
    )

    denominator = np.abs(y_true).sum()

    if denominator == 0:
        wape = np.nan
    else:
        wape = (
            np.abs(y_true - y_pred).sum()
            / denominator
        ) * 100

    return {
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2),
        "WAPE": (
            round(wape, 2)
            if not np.isnan(wape)
            else np.nan
        ),
    }


def run_backtest():
    print()
    print("======================================")
    print("       RES WALK-FORWARD BACKTEST")
    print("======================================")
    print()

    df = build_model_data()

    available_end = (
        df["timestamp"].max()
        + pd.Timedelta(hours=1)
    )

    backtest_end = min(
        BACKTEST_END,
        available_end,
    )

    month_starts = pd.date_range(
        start=BACKTEST_START,
        end=backtest_end,
        freq="MS",
        inclusive="left",
    )

    results = []

    for test_start in month_starts:
        test_end = (
            test_start
            + pd.offsets.MonthBegin(1)
        )

        # Walk-forward evaluation:
        # train only on observations before the test month.
        train = df[
            df["timestamp"] < test_start
        ].copy()

        test = df[
            (df["timestamp"] >= test_start)
            & (df["timestamp"] < test_end)
        ].copy()

        if train.empty:
            print(
                f"Skipping {test_start:%Y-%m}: no training data."
            )
            continue

        if test.empty:
            print(
                f"Skipping {test_start:%Y-%m}: no test data."
            )
            continue

        model = build_model()

        X_train = train[FEATURE_COLS]
        y_train = train["target"]

        X_test = test[FEATURE_COLS]
        y_test = test["target"]

        model.fit(
            X_train,
            y_train,
        )

        predictions = model.predict(X_test)

        # RES production cannot be negative.
        predictions = np.maximum(
            predictions,
            0.0,
        )

        metrics = evaluate(
            y_true=y_test,
            y_pred=predictions,
        )

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
        raise ValueError(
            "RES backtest produced no results."
        )

    print()
    print("======================================")
    print("          RES BACKTEST SUMMARY")
    print("======================================")
    print(
        f"Average MAE:  "
        f"{results_df['MAE'].mean():.2f} MWh"
    )
    print(
        f"Average RMSE: "
        f"{results_df['RMSE'].mean():.2f} MWh"
    )
    print(
        f"Average WAPE: "
        f"{results_df['WAPE'].mean():.2f}%"
    )

    best = results_df.loc[
        results_df["WAPE"].idxmin()
    ]

    worst = results_df.loc[
        results_df["WAPE"].idxmax()
    ]

    print(
        f"Best month: "
        f"{best['month']} "
        f"({best['WAPE']:.2f}% WAPE)"
    )

    print(
        f"Worst month: "
        f"{worst['month']} "
        f"({worst['WAPE']:.2f}% WAPE)"
    )

    RESULTS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_df.to_csv(
        RESULTS_PATH,
        index=False,
    )

    print()
    print(
        f"Results saved to: {RESULTS_PATH}"
    )

    return results_df


def main():
    run_backtest()


if __name__ == "__main__":
    main()
