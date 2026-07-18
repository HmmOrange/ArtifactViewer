#!/usr/bin/env python3
"""Download the shared Google Drive folder into ./data.

Install the small runtime dependencies once with:
    python -m pip install gdown rich

Then run:
    python scripts/download_data.py
"""

from __future__ import annotations

import argparse
import importlib
import os
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from pathlib import Path


FOLDER_URL = "https://drive.google.com/drive/folders/1fN3uyZQtuLmIdp284dMDV9Omg0shMT5e?usp=sharing"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download the shared Google Drive folder into a local directory."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data"),
        help="Destination directory (default: ./data)",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Do not reuse partial files from an earlier interrupted download.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Concurrent downloads (default: 8)",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.workers < 1:
        print("--workers must be at least 1", file=sys.stderr)
        return 2

    # Rich uses Unicode progress glyphs; force a consistent Windows encoding.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")

    try:
        import gdown
        import requests
        from rich.console import Console
        from rich.progress import (
            BarColumn,
            DownloadColumn,
            Progress,
            SpinnerColumn,
            TextColumn,
            TimeElapsedColumn,
            TimeRemainingColumn,
            TransferSpeedColumn,
        )
        from requests.exceptions import RequestException
    except ImportError as exc:
        print(
            f"Missing dependency: {exc.name}. Install it with "
            "'python -m pip install gdown rich'.",
            file=sys.stderr,
        )
        return 2

    console = Console()
    destination = args.output.expanduser()
    destination.mkdir(parents=True, exist_ok=True)

    console.print(f"[bold cyan]Downloading dataset[/bold cyan] -> {destination.resolve()}")

    try:
        folder_api = importlib.import_module("gdown.download_folder")
        user_agent = (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 Chrome/126 Safari/537.36"
        )
        session, _ = folder_api._get_session(
            proxy=None,
            use_cookies=False,
            user_agent=user_agent,
        )
        session.get = partial(session.get, timeout=(15, 60))

        progress = Progress(
            SpinnerColumn(style="cyan"),
            TextColumn("[bold]{task.description}"),
            BarColumn(bar_width=None, style="bright_black", complete_style="cyan"),
            DownloadColumn(),
            TransferSpeedColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console,
            expand=True,
        )

        completed = 0
        discovered = 0
        aggregate_done = 0
        aggregate_total = 0
        file_progress: dict[str, tuple[int, int]] = {}
        progress_lock = threading.Lock()
        thread_local = threading.local()
        folder_id = folder_api._extract_folder_id(FOLDER_URL)
        pending_folders = [(folder_id, Path())]

        def worker_session():
            if not hasattr(thread_local, "session"):
                thread_local.session = requests.Session()
                thread_local.session.headers["User-Agent"] = user_agent
            return thread_local.session

        try:
            with progress:
                activity = progress.add_task("Downloading | 0/0 files", total=None)

                def update_file(
                    file_id: str,
                    bytes_done: int,
                    total: int | None,
                ) -> None:
                    nonlocal aggregate_done, aggregate_total
                    with progress_lock:
                        previous_done, previous_total = file_progress.get(file_id, (0, 0))
                        known_total = total or previous_total
                        aggregate_done += bytes_done - previous_done
                        aggregate_total += known_total - previous_total
                        file_progress[file_id] = (bytes_done, known_total)
                        progress.update(
                            activity,
                            description=f"Downloading | {completed}/{discovered} files",
                            completed=aggregate_done,
                            total=aggregate_total or None,
                        )

                def download_one(file_id: str, target: Path) -> None:
                    nonlocal completed
                    target.parent.mkdir(parents=True, exist_ok=True)

                    if target.exists() and not args.fresh:
                        size = target.stat().st_size
                        update_file(file_id, size, size)
                    else:
                        for attempt in range(1, 4):
                            try:
                                part_path = target.with_name(target.name + ".part")
                                initial = (
                                    part_path.stat().st_size
                                    if part_path.exists()
                                    and not args.fresh
                                    and target.suffix
                                    else 0
                                )
                                headers = (
                                    {"Range": f"bytes={initial}-"} if initial else {}
                                )
                                response = worker_session().get(
                                    "https://drive.google.com/uc",
                                    params={"export": "download", "id": file_id},
                                    headers=headers,
                                    stream=True,
                                    timeout=(15, 60),
                                )
                                response.raise_for_status()

                                content_type = response.headers.get("Content-Type", "")
                                if "text/html" in content_type:
                                    response.close()
                                    raise RuntimeError(
                                        "Google Drive did not return a file"
                                    )

                                if not target.suffix:
                                    disposition = response.headers.get(
                                        "Content-Disposition", ""
                                    )
                                    match = re.search(r'filename="?([^";]+)', disposition)
                                    if match:
                                        target = target.parent / match.group(1)
                                        part_path = target.with_name(target.name + ".part")
                                        if target.exists() and not args.fresh:
                                            response.close()
                                            size = target.stat().st_size
                                            update_file(file_id, size, size)
                                            break

                                if response.status_code != 206:
                                    initial = 0
                                total_header = response.headers.get("Content-Length")
                                total = (
                                    int(total_header) + initial if total_header else None
                                )
                                mode = "ab" if initial else "wb"
                                with response, part_path.open(mode) as output_file:
                                    bytes_done = initial
                                    update_file(file_id, bytes_done, total)
                                    for chunk in response.iter_content(1024 * 1024):
                                        if not chunk:
                                            continue
                                        output_file.write(chunk)
                                        bytes_done += len(chunk)
                                        update_file(file_id, bytes_done, total)

                                os.replace(part_path, target)
                                break
                            except RequestException:
                                if attempt == 3:
                                    raise

                    with progress_lock:
                        completed += 1
                        progress.update(
                            activity,
                            description=f"Downloading | {completed}/{discovered} files",
                        )

                futures = []
                with ThreadPoolExecutor(max_workers=args.workers) as executor:
                    # Discovery and transfers overlap; filenames remain hidden.
                    while pending_folders:
                        current_folder_id, relative_dir = pending_folders.pop()
                        _, children = folder_api._parse_embedded_folder_view(
                            session,
                            current_folder_id,
                        )

                        child_folders = []
                        for child_id, child_name, child_type in children:
                            safe_name = folder_api._sanitize_filename(child_name)
                            child_path = relative_dir / safe_name
                            if child_type == folder_api._GoogleDriveFile.TYPE_FOLDER:
                                child_folders.append((child_id, child_path))
                                continue

                            with progress_lock:
                                discovered += 1
                            futures.append(
                                executor.submit(
                                    download_one,
                                    child_id,
                                    destination / child_path,
                                )
                            )

                        pending_folders.extend(reversed(child_folders))

                    for future in as_completed(futures):
                        future.result()
        finally:
            session.close()
    except KeyboardInterrupt:
        console.print("\n[yellow]Download interrupted.[/yellow] Re-run to resume.")
        return 130
    except Exception as exc:  # gdown exposes several download-related exception types.
        console.print(f"\n[bold red]Download failed:[/bold red] {exc}")
        return 1

    console.print(f"[bold green]Done.[/bold green]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
