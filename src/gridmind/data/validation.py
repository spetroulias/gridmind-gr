import pandas as pd


def validate_system_load(
    df: pd.DataFrame,
) -> None:
    required_columns = {
        "date",
        "period",
        "net_load_mwh",
        "crete_flow_mwh",
    }

    missing = (
        required_columns
        - set(df.columns)
    )

    if missing:
        raise ValueError(
            f"Missing columns: {missing}"
        )

    if df["date"].isna().any():
        raise ValueError(
            "Missing dates."
        )

    if df["period"].isna().any():
        raise ValueError(
            "Missing periods."
        )

    if not df["period"].between(
        1,
        25,
    ).all():
        raise ValueError(
            "Invalid period."
        )

    if df["net_load_mwh"].isna().any():
        raise ValueError(
            "Missing system load values."
        )

    if df["crete_flow_mwh"].isna().any():
        raise ValueError(
            "Missing Crete flow values."
        )

    if df.duplicated(
        subset=["date", "period"]
    ).any():
        raise ValueError(
            "Duplicate date-period rows."
        )


def validate_res(
    df: pd.DataFrame,
) -> None:
    required_columns = {
        "date",
        "period",
        "res_mwh",
    }

    missing = (
        required_columns
        - set(df.columns)
    )

    if missing:
        raise ValueError(
            f"Missing RES columns: {missing}"
        )

    if df["date"].isna().any():
        raise ValueError(
            "Missing RES dates."
        )

    if df["period"].isna().any():
        raise ValueError(
            "Missing RES periods."
        )

    if not df["period"].between(
        1,
        25,
    ).all():
        raise ValueError(
            "Invalid RES period."
        )

    if df["res_mwh"].isna().any():
        raise ValueError(
            "Missing RES values."
        )

    if df.duplicated(
        subset=["date", "period"]
    ).any():
        raise ValueError(
            "Duplicate RES date-period rows."
        )


def validate_generation(
    df: pd.DataFrame,
) -> None:
    required_columns = {
        "date",
        "period",
        "unit_name",
        "technology",
        "production_mwh",
    }

    missing = (
        required_columns
        - set(df.columns)
    )

    if missing:
        raise ValueError(
            f"Missing generation columns: {missing}"
        )

    if df.empty:
        raise ValueError(
            "Generation dataset is empty."
        )

    if df["date"].isna().any():
        raise ValueError(
            "Missing generation dates."
        )

    if df["period"].isna().any():
        raise ValueError(
            "Missing generation periods."
        )

    if not df["period"].between(
        1,
        25,
    ).all():
        raise ValueError(
            "Invalid generation period."
        )

    if df["unit_name"].isna().any():
        raise ValueError(
            "Missing generation unit names."
        )

    if (
        df["unit_name"]
        .astype(str)
        .str.strip()
        .eq("")
        .any()
    ):
        raise ValueError(
            "Empty generation unit names."
        )

    allowed_technologies = {
        "lignite",
        "petroleum",
        "natural_gas",
        "hydro",
        "res",
    }

    invalid_technologies = (
        set(df["technology"].unique())
        - allowed_technologies
    )

    if invalid_technologies:
        raise ValueError(
            "Invalid generation technologies: "
            f"{invalid_technologies}"
        )

    if df["production_mwh"].isna().any():
        raise ValueError(
            "Missing generation values."
        )

    if df.duplicated(
        subset=[
            "date",
            "period",
            "unit_name",
        ]
    ).any():
        raise ValueError(
            "Duplicate generation "
            "date-period-unit rows."
        )