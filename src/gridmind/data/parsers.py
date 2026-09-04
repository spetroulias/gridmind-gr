from pathlib import Path

import pandas as pd


def parse_system_load(file_path: str | Path) -> pd.DataFrame:
    df = pd.read_excel(file_path, header=None)

    net_load_date = pd.to_datetime(
        df.iloc[2, 0],
        format="%d-%m-%Y",
    )

    net_load_values = df.iloc[2, 1:26].astype(float)

    crete_flow_date = pd.to_datetime(
        df.iloc[8, 0],
        format="%d-%m-%Y",
    )

    crete_flow_values = df.iloc[8, 1:26].astype(float)

    if net_load_date != crete_flow_date:
        raise ValueError(
            "Dates in ADMIE data blocks do not match."
        )

    result = pd.DataFrame(
        {
            "date": net_load_date,
            "period": range(1, 26),
            "net_load_mwh": net_load_values.to_numpy(),
            "crete_flow_mwh": crete_flow_values.to_numpy(),
        }
    )

    return result


def parse_system_load_files(
    file_paths: list[Path],
) -> pd.DataFrame:
    if not file_paths:
        raise ValueError("No ADMIE files were provided.")

    dataframes = []

    for file_path in file_paths:
        df = parse_system_load(file_path)
        dataframes.append(df)

    result = pd.concat(
        dataframes,
        ignore_index=True,
    )

    return result