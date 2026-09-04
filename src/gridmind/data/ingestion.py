import argparse
from pathlib import Path

import pandas as pd

from gridmind.data.admie_client import AdmieClient
from gridmind.data.parsers import parse_system_load_files
from gridmind.data.validation import validate_system_load


SYSTEM_LOAD_CATEGORY = "RealTimeSCADASystemLoad"


def ingest_system_load(
    start_date: str,
    end_date: str,
    raw_folder: str = "data/raw",
    processed_file: str = "data/processed/system_load.csv",
) -> pd.DataFrame:
    client = AdmieClient()

    print(
        f"Fetching ADMIE System Load files "
        f"from {start_date} to {end_date}..."
    )

    file_paths = client.download_files(
        start_date=start_date,
        end_date=end_date,
        file_category=SYSTEM_LOAD_CATEGORY,
        destination_folder=raw_folder,
    )

    print(f"Found {len(file_paths)} files.")

    df = parse_system_load_files(file_paths)

    df = df.sort_values(
        ["date", "period"]
    ).reset_index(drop=True)

    validate_system_load(df)

    output_path = Path(processed_file)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        output_path,
        index=False,
    )

    print("Validation passed.")
    print(f"Rows: {len(df)}")
    print(f"Saved processed data to: {output_path}")

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ingest ADMIE System Load data."
    )

    parser.add_argument(
        "--start-date",
        required=True,
    )

    parser.add_argument(
        "--end-date",
        required=True,
    )

    args = parser.parse_args()

    ingest_system_load(
        start_date=args.start_date,
        end_date=args.end_date,
    )