import argparse
from pathlib import Path

import pandas as pd

from gridmind.data.admie_client import AdmieClient
from gridmind.data.database import save_system_load
from gridmind.data.parsers import parse_system_load_files
from gridmind.data.validation import validate_system_load


SYSTEM_LOAD_CATEGORY = (
    "RealTimeSCADASystemLoad"
)


def ingest_system_load(
    start_date: str,
    end_date: str,
    raw_folder: str = "data/raw",
    processed_file: str = (
        "data/processed/system_load.csv"
    ),
) -> pd.DataFrame:

    client = AdmieClient()

    print(
        f"Fetching System Load "
        f"{start_date} → {end_date}"
    )

    file_paths = client.download_files(
        start_date=start_date,
        end_date=end_date,
        file_category=SYSTEM_LOAD_CATEGORY,
        destination_folder=raw_folder,
    )

    print(
        f"{len(file_paths)} files found."
    )

    df = parse_system_load_files(
        file_paths
    )

    df = (
        df.sort_values(
            ["date", "period"]
        )
        .reset_index(drop=True)
    )

    validate_system_load(df)

    output_path = Path(
        processed_file
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        output_path,
        index=False,
    )

    save_system_load(df)

    print("Validation passed.")
    print(f"{len(df)} rows processed.")
    print(
        f"CSV saved to {output_path}"
    )
    print(
        "Data saved to PostgreSQL."
    )

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

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