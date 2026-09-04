import pandas as pd


def validate_system_load(df: pd.DataFrame) -> None:
    required_columns = {
        "date",
        "period",
        "net_load_mwh",
        "crete_flow_mwh",
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    if df["date"].isna().any():
        raise ValueError("Missing dates found.")

    if df["period"].isna().any():
        raise ValueError("Missing periods found.")

    if not df["period"].between(1, 25).all():
        raise ValueError("Period must be between 1 and 25.")

    if df["net_load_mwh"].isna().any():
        raise ValueError("Missing net load values found.")

    if df["crete_flow_mwh"].isna().any():
        raise ValueError("Missing Crete flow values found.")

    if df.duplicated(subset=["date", "period"]).any():
        raise ValueError("Duplicate date-period rows found.")