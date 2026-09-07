import argparse

import pandas as pd

from gridmind.data.admie_client import AdmieClient
from gridmind.data.database import (
    save_generation,
    save_res,
    save_system_load,
)
from gridmind.data.parsers import (
    parse_generation_files,
    parse_res_files,
    parse_system_load_files,
)
from gridmind.data.validation import (
    validate_generation,
    validate_res,
    validate_system_load,
)


SYSTEM_LOAD_CATEGORY = "RealTimeSCADASystemLoad"
RES_CATEGORY = "RealTimeSCADARES"
GENERATION_CATEGORY = "SystemRealizationSCADA"


def ingest_system_load(
    start_date: str,
    end_date: str,
    raw_folder: str = "data/raw",
) -> pd.DataFrame:
    client = AdmieClient()

    print(f"Fetching System Load {start_date} -> {end_date}")

    file_paths = client.download_files(
        start_date=start_date,
        end_date=end_date,
        file_category=SYSTEM_LOAD_CATEGORY,
        destination_folder=raw_folder,
    )

    print(f"{len(file_paths)} files found.")

    df = (
        parse_system_load_files(file_paths)
        .sort_values(["date", "period"])
        .reset_index(drop=True)
    )

    validate_system_load(df)
    print("System Load validation passed.")

    save_system_load(df)
    print(f"{len(df)} System Load rows saved to PostgreSQL.")

    return df


def ingest_res(
    start_date: str,
    end_date: str,
    raw_folder: str = "data/raw",
) -> pd.DataFrame:
    client = AdmieClient()

    print(f"Fetching RES {start_date} -> {end_date}")

    file_paths = client.download_files(
        start_date=start_date,
        end_date=end_date,
        file_category=RES_CATEGORY,
        destination_folder=raw_folder,
    )

    print(f"{len(file_paths)} files found.")

    df = (
        parse_res_files(file_paths)
        .sort_values(["date", "period"])
        .reset_index(drop=True)
    )

    validate_res(df)
    print("RES validation passed.")

    save_res(df)
    print(f"{len(df)} RES rows saved to PostgreSQL.")

    return df


def ingest_generation(
    start_date: str,
    end_date: str,
    raw_folder: str = "data/raw",
) -> pd.DataFrame:
    client = AdmieClient()

    print(f"Fetching Generation {start_date} -> {end_date}")

    file_paths = client.download_files(
        start_date=start_date,
        end_date=end_date,
        file_category=GENERATION_CATEGORY,
        destination_folder=raw_folder,
    )

    print(f"{len(file_paths)} files found.")

    df = (
        parse_generation_files(file_paths)
        .sort_values(
            ["date", "period", "technology", "unit_name"]
        )
        .reset_index(drop=True)
    )

    validate_generation(df)
    print("Generation validation passed.")

    save_generation(df)
    print(f"{len(df)} Generation rows saved to PostgreSQL.")

    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest ADMIE energy datasets."
    )

    parser.add_argument(
        "dataset",
        choices=["system-load", "res", "generation"],
        help="Dataset to ingest.",
    )
    parser.add_argument(
        "--start-date",
        required=True,
        help="Start date in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--end-date",
        required=True,
        help="End date in YYYY-MM-DD format.",
    )

    args = parser.parse_args()

    if args.dataset == "system-load":
        ingest_system_load(args.start_date, args.end_date)
    elif args.dataset == "res":
        ingest_res(args.start_date, args.end_date)
    else:
        ingest_generation(args.start_date, args.end_date)


if __name__ == "__main__":
    main()
