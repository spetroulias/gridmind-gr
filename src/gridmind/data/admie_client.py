from pathlib import Path
from typing import Any

import requests


class AdmieClient:
    BASE_URL = "https://www.admie.gr"

    def __init__(self, timeout: int = 30):
        self.timeout = timeout

    def get_files(
        self,
        start_date: str,
        end_date: str,
        file_category: str,
    ) -> list[dict[str, Any]]:
        url = f"{self.BASE_URL}/getOperationMarketFilewRange"

        params = {
            "dateStart": start_date,
            "dateEnd": end_date,
            "FileCategory": file_category,
        }

        response = requests.get(
            url,
            params=params,
            timeout=self.timeout,
        )

        response.raise_for_status()

        return response.json()

    def download_file(
        self,
        file_url: str,
        destination_folder: str,
    ) -> Path:
        response = requests.get(
            file_url,
            timeout=self.timeout,
        )

        response.raise_for_status()

        filename = file_url.split("/")[-1]

        folder = Path(destination_folder)
        folder.mkdir(parents=True, exist_ok=True)

        file_path = folder / filename

        file_path.write_bytes(response.content)

        return file_path

    def download_files(
        self,
        start_date: str,
        end_date: str,
        file_category: str,
        destination_folder: str,
    ) -> list[Path]:

        files = self.get_files(
            start_date=start_date,
            end_date=end_date,
            file_category=file_category,
        )

        downloaded_files = []

        for file in files:
            file_path = self.download_file(
                file_url=file["file_path"],
                destination_folder=destination_folder,
            )

            downloaded_files.append(file_path)

        return downloaded_files


if __name__ == "__main__":
    client = AdmieClient()

    downloaded_files = client.download_files(
        start_date="2026-01-15",
        end_date="2026-01-20",
        file_category="RealTimeSCADASystemLoad",
        destination_folder="data/raw",
    )

    for file_path in downloaded_files:
        print(f"Saved to: {file_path}")