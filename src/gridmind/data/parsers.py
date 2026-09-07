from pathlib import Path

import pandas as pd


GENERATION_TECHNOLOGY_HEADERS = {
    "ΛΙΓΝΙΤΙΚΕΣ ΜΟΝΑΔΕΣ": "lignite",
    "ΠΕΤΡΕΛΑΙΚΕΣ ΜΟΝΑΔΕΣ": "petroleum",
    "ΜΟΝΑΔΕΣ Φ. ΑΕΡΙΟΥ": "natural_gas",
    "ΥΔΡΟΗΛΕΚΤΡΙΚΕΣ ΜΟΝΑΔΕΣ": "hydro",
    "ΑΝΑΝΕΩΣΙΜΑ": "res",
}


GENERATION_TOTAL_ROWS = {
    "TOTAL LIGNITE",
    "ΣΥΝΟΛΟ ΠΕΤΡΕΛΑΙΚΩΝ",
    "TOTAL GAS",
    "TOTAL HYDRO",
    "TOTAL RES",
}


def parse_system_load(
    file_path: str | Path,
) -> pd.DataFrame:
    df = pd.read_excel(
        file_path,
        header=None,
    )

    net_load_date = pd.to_datetime(
        df.iloc[2, 0],
        format="%d-%m-%Y",
    )

    net_load_values = (
        df.iloc[2, 1:26]
        .astype(float)
    )

    crete_flow_date = pd.to_datetime(
        df.iloc[8, 0],
        format="%d-%m-%Y",
    )

    crete_flow_values = (
        df.iloc[8, 1:26]
        .astype(float)
    )

    if net_load_date != crete_flow_date:
        raise ValueError(
            "Dates in ADMIE blocks do not match."
        )

    return pd.DataFrame(
        {
            "date": net_load_date,
            "period": range(1, 26),
            "net_load_mwh": net_load_values.to_numpy(),
            "crete_flow_mwh": crete_flow_values.to_numpy(),
        }
    )


def parse_system_load_files(
    file_paths: list[Path],
) -> pd.DataFrame:
    if not file_paths:
        raise ValueError(
            "No ADMIE System Load files found."
        )

    dataframes = [
        parse_system_load(file_path)
        for file_path in file_paths
    ]

    return pd.concat(
        dataframes,
        ignore_index=True,
    )


def parse_res(
    file_path: str | Path,
) -> pd.DataFrame:
    df = pd.read_excel(
        file_path,
        header=None,
    )

    date = pd.to_datetime(
        df.iloc[2, 0],
        format="%d-%m-%Y",
    )

    res_values = (
        df.iloc[2, 1:26]
        .astype(float)
    )

    return pd.DataFrame(
        {
            "date": date,
            "period": range(1, 26),
            "res_mwh": res_values.to_numpy(),
        }
    )


def parse_res_files(
    file_paths: list[Path],
) -> pd.DataFrame:
    if not file_paths:
        raise ValueError(
            "No ADMIE RES files found."
        )

    dataframes = [
        parse_res(file_path)
        for file_path in file_paths
    ]

    return pd.concat(
        dataframes,
        ignore_index=True,
    )


def _extract_generation_date(
    file_path: str | Path,
) -> pd.Timestamp:
    """
    ADMIE SystemRealizationSCADA filenames follow:

        YYYYMMDD_SystemRealizationSCADA_01.xls

    Using the filename is considerably more robust than parsing
    the Greek human-readable date in the worksheet title.
    """

    filename = Path(file_path).name

    date_string = filename[:8]

    try:
        return pd.to_datetime(
            date_string,
            format="%Y%m%d",
        )
    except ValueError as exc:
        raise ValueError(
            f"Could not determine date from "
            f"generation filename: {filename}"
        ) from exc


def _get_generation_period_columns(
    header_row: pd.Series,
) -> list[tuple[int, int]]:
    """
    Identify period columns from a technology header row.

    The inspected ADMIE workbook contains hourly columns
    labelled 1..24 followed by SUM.

    We detect them instead of assuming a fixed number so that
    DST files can preserve an additional period if ADMIE
    publishes one.
    """

    period_columns = []

    for column_index in range(
        2,
        len(header_row),
    ):
        value = header_row.iloc[column_index]

        if pd.isna(value):
            continue

        if str(value).strip().upper() == "SUM":
            break

        try:
            period = int(float(value))
        except (TypeError, ValueError):
            continue

        if 1 <= period <= 25:
            period_columns.append(
                (
                    column_index,
                    period,
                )
            )

    if not period_columns:
        raise ValueError(
            "Could not detect generation period columns."
        )

    return period_columns


def parse_generation(
    file_path: str | Path,
) -> pd.DataFrame:
    """
    Parse ADMIE SystemRealizationSCADA production-unit data.

    Clean schema:

        date
        period
        unit_name
        technology
        production_mwh

    Aggregate TOTAL rows are intentionally excluded because
    they can be derived from the individual units.
    """

    df = pd.read_excel(
        file_path,
        sheet_name="System_Production",
        header=None,
    )

    date = _extract_generation_date(
        file_path
    )

    records = []

    current_technology = None
    period_columns = None

    for _, row in df.iterrows():
        raw_name = row.iloc[1]

        if pd.isna(raw_name):
            continue

        name = str(raw_name).strip()

        if name in GENERATION_TECHNOLOGY_HEADERS:
            current_technology = (
                GENERATION_TECHNOLOGY_HEADERS[name]
            )

            period_columns = (
                _get_generation_period_columns(
                    row
                )
            )

            continue

        if current_technology is None:
            continue

        if name in GENERATION_TOTAL_ROWS:
            current_technology = None
            period_columns = None
            continue

        if period_columns is None:
            continue

        for column_index, period in period_columns:
            value = row.iloc[column_index]

            if pd.isna(value):
                raise ValueError(
                    f"Missing generation value in "
                    f"{Path(file_path).name}: "
                    f"unit={name}, period={period}"
                )

            try:
                production_mwh = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid generation value in "
                    f"{Path(file_path).name}: "
                    f"unit={name}, "
                    f"period={period}, "
                    f"value={value!r}"
                ) from exc

            records.append(
                {
                    "date": date,
                    "period": period,
                    "unit_name": name,
                    "technology": current_technology,
                    "production_mwh": production_mwh,
                }
            )

    if not records:
        raise ValueError(
            f"No generation records found in "
            f"{Path(file_path).name}"
        )

    return pd.DataFrame(
        records
    )


def parse_generation_files(
    file_paths: list[Path],
) -> pd.DataFrame:
    if not file_paths:
        raise ValueError(
            "No ADMIE generation files found."
        )

    dataframes = [
        parse_generation(file_path)
        for file_path in file_paths
    ]

    return pd.concat(
        dataframes,
        ignore_index=True,
    )