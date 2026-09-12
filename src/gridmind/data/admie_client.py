
from pathlib import Path
from urllib.parse import urljoin, urlparse
from datetime import date
import tempfile
import os
from time import sleep
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


class AdmieClient:
    BASE_URL = "https://www.admie.gr"

    def __init__(
        self,
        timeout: int = 120,
        max_retries: int = 5,
        backoff_factor: float = 2.0,
    ):
        self.timeout = timeout

        retry_strategy = Retry(
            total=max_retries,
            connect=max_retries,
            read=max_retries,
            status=max_retries,
            backoff_factor=backoff_factor,
            status_forcelist=[
                429,
                500,
                502,
                503,
                504,
            ],
            allowed_methods=["GET"],
            raise_on_status=False,
        )

        adapter = HTTPAdapter(
            max_retries=retry_strategy
        )

        self.session = requests.Session()

        self.session.mount(
            "https://",
            adapter,
        )

        self.session.mount(
            "http://",
            adapter,
        )

    def get_files(
        self,
        start_date: str,
        end_date: str,
        file_category: str,
    ) -> list[dict[str, Any]]:
        if date.fromisoformat(start_date) > date.fromisoformat(end_date):
            raise ValueError("start_date must not be after end_date")
        url = (
            f"{self.BASE_URL}/"
            "getOperationMarketFilewRange"
        )

        params = {
            "dateStart": start_date,
            "dateEnd": end_date,
            "FileCategory": file_category,
        }

        response = self.session.get(
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
        file_url = urljoin(self.BASE_URL, file_url)
        filename = Path(urlparse(file_url).path).name
        if not filename or filename in {".", ".."}:
            raise ValueError("Invalid ADMIE file URL")

        folder = Path(destination_folder)

        folder.mkdir(
            parents=True,
            exist_ok=True,
        )

        file_path = folder / filename

        if file_path.exists() and file_path.stat().st_size > 0:
            return file_path

        response = self.session.get(
            file_url,
            timeout=self.timeout,
        )

        response.raise_for_status()

        if not response.content:
            raise ValueError("ADMIE returned an empty file")
        with tempfile.NamedTemporaryFile(dir=folder, delete=False) as temporary:
            temporary.write(response.content)
            temporary_path = temporary.name
        os.replace(temporary_path, file_path)

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

        total_files = len(files)

        print(
            f"{total_files} files available "
            f"from ADMIE."
        )

        file_paths = []

        for index, file in enumerate(
            files,
            start=1,
        ):
            filename = (
                file["file_path"]
                .split("/")[-1]
            )

            destination_path = (
                Path(destination_folder)
                / filename
            )

            if destination_path.exists():
                print(
                    f"[{index}/{total_files}] "
                    f"Already exists: {filename}"
                )
            else:
                print(
                    f"[{index}/{total_files}] "
                    f"Downloading: {filename}"
                )

            try:
                file_path = self.download_file(
                    file_url=file["file_path"],
                    destination_folder=destination_folder,
                )

            except requests.RequestException as exc:
                print(
                    "\nDownload failed after retries:"
                )
                print(filename)
                print(exc)

                raise

            file_paths.append(
                file_path
            )

            # Small delay to avoid hammering
            # the ADMIE server.
            sleep(0.1)

        return file_paths

