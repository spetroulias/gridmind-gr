import pandas as pd
import numpy as np


VALID_TECHNOLOGIES = {
    "lignite",
    "petroleum",
    "natural_gas",
    "hydro",
    "res",
}


def _require_columns(df: pd.DataFrame, required: set[str], label: str) -> None:
    if df.empty:
        raise ValueError(f"{label} dataset is empty.")
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Missing {label} columns: {missing}")
    if pd.to_datetime(df["date"], errors="coerce").isna().any():
        raise ValueError(f"Invalid {label} dates.")
    for column in required:
        if column.endswith("_mwh"):
            if not pd.api.types.is_numeric_dtype(df[column]) or not np.isfinite(df[column]).all():
                raise ValueError(f"Missing or non-finite {label} values in {column}.")


def _validate_periods(df: pd.DataFrame, label: str) -> None:
    if df["period"].isna().any():
        raise ValueError(f"Missing {label} periods.")

    if not pd.api.types.is_numeric_dtype(df["period"]) or not df["period"].between(1, 25).all() or not df["period"].mod(1).eq(0).all():
        raise ValueError(f"Invalid {label} period.")


def validate_system_load(df: pd.DataFrame) -> None:
    required = {
        "date",
        "period",
        "net_load_mwh",
        "crete_flow_mwh",
    }
    _require_columns(df, required, "system load")

    if df["date"].isna().any():
        raise ValueError("Missing system load dates.")

    _validate_periods(df, "system load")

    if df["net_load_mwh"].isna().any():
        raise ValueError("Missing system load values.")

    if df["crete_flow_mwh"].isna().any():
        raise ValueError("Missing Crete flow values.")

    if df.duplicated(subset=["date", "period"]).any():
        raise ValueError("Duplicate system load date-period rows.")


def validate_res(df: pd.DataFrame) -> None:
    required = {"date", "period", "res_mwh"}
    _require_columns(df, required, "RES")

    if df["date"].isna().any():
        raise ValueError("Missing RES dates.")

    _validate_periods(df, "RES")

    if df["res_mwh"].isna().any():
        raise ValueError("Missing RES values.")

    if df.duplicated(subset=["date", "period"]).any():
        raise ValueError("Duplicate RES date-period rows.")


def validate_generation(df: pd.DataFrame) -> None:
    required = {
        "date",
        "period",
        "unit_name",
        "technology",
        "production_mwh",
    }
    _require_columns(df, required, "generation")

    if df.empty:
        raise ValueError("Generation dataset is empty.")

    if df["date"].isna().any():
        raise ValueError("Missing generation dates.")

    _validate_periods(df, "generation")

    if df["unit_name"].isna().any():
        raise ValueError("Missing generation unit names.")

    if df["unit_name"].astype(str).str.strip().eq("").any():
        raise ValueError("Empty generation unit names.")

    invalid_technologies = set(df["technology"].unique()) - VALID_TECHNOLOGIES

    if invalid_technologies:
        raise ValueError(
            f"Invalid generation technologies: {invalid_technologies}"
        )

    if df["production_mwh"].isna().any():
        raise ValueError("Missing generation values.")

    if df.duplicated(
        subset=["date", "period", "unit_name"]
    ).any():
        raise ValueError("Duplicate generation date-period-unit rows.")
