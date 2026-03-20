"""
Scraper for sensors.AFRICA Air Quality Archive - Nairobi
Fetches CSV resources from the CKAN API every 30 minutes
and saves new / updated files to data/raw_data/.
"""
import json
import os
import sys
import time
import logging
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

import re
import shutil

import requests

# ---------------------------------------------
# CONFIGURATION
# ---------------------------------------------

CKAN_API_URL = (
    "https://open.africa/api/3/action/package_show"
    "?id=sensorsafrica-airquality-archive-nairobi"
)
RAW_DATA_DIR = Path(__file__).resolve().parent / "data" / "raw_data"
LOG_DIR = Path(__file__).resolve().parent / "logs"
METADATA_PATH = RAW_DATA_DIR / "_metadata.json"
SCRAPE_INTERVAL_SECONDS = 20 * 60  # 20 minutes
REQUEST_TIMEOUT = 120  # seconds per download
CHUNK_SIZE = 8192  # streaming download chunk

# ---------------------------------------------
# LOGGING
# ---------------------------------------------

LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_DIR / "scraper.log", encoding="utf-8"),
    ],
)
log = logging.getLogger("scraper")


# ---------------------------------------------
# HELPERS
# ---------------------------------------------

def _filename_from_url(url: str) -> str:
    """Extract the filename from a CKAN download URL."""
    return Path(urlsplit(url).path).name


def _semantic_filename(resource_name: str) -> str:
    """
    Derive a clean filename from a CKAN resource name.
    e.g. 'January 2023 Sensor Data Archive' -> 'january_2023_sensor_data_archive.csv'
    """
    name = resource_name.strip().lower()
    # Replace any non-alphanumeric char (except spaces) with nothing,
    # then replace spaces with underscores
    name = re.sub(r"[^a-z0-9 ]", "", name)
    name = re.sub(r"\s+", "_", name).strip("_")
    if not name:
        return ""
    return name + ".csv"


def _load_metadata() -> dict:
    """Load the download-tracking metadata (resource_id -> last_modified)."""
    if METADATA_PATH.exists():
        return json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    return {}


def _save_metadata(meta: dict) -> None:
    """Persist the download-tracking metadata to disk."""
    METADATA_PATH.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def _fetch_resource_list() -> list[dict]:
    """Call the CKAN API and return the list of CSV resources."""
    log.info("Fetching resource list from CKAN API ...")
    resp = requests.get(CKAN_API_URL, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    data = resp.json()
    if not data.get("success"):
        raise RuntimeError(f"CKAN API returned success=false: {data}")
    resources = data["result"]["resources"]
    # Keep only CSV resources
    csv_resources = [r for r in resources if r.get("format", "").upper() == "CSV"]
    log.info("Found %d CSV resources on the server.", len(csv_resources))
    return csv_resources


def _download_resource(url: str, dest: Path) -> bool:
    """
    Stream-download a single CSV resource to *dest*.
    Returns True on success, False on 404 / other expected failures.
    """
    tmp_dest = dest.with_suffix(".tmp")
    try:
        with requests.get(url, stream=True, timeout=REQUEST_TIMEOUT) as resp:
            if resp.status_code == 404:
                log.warning("  [!] 404 Not Found -- skipping: %s", url)
                return False
            resp.raise_for_status()

            # Verify we actually got CSV-like content (not an HTML error page)
            content_type = resp.headers.get("Content-Type", "")
            if "text/html" in content_type:
                log.warning("  [!] Received HTML instead of CSV -- skipping: %s", url)
                return False

            with open(tmp_dest, "wb") as f:
                for chunk in resp.iter_content(chunk_size=CHUNK_SIZE):
                    f.write(chunk)

            # Only keep files with actual data (> 100 bytes)
            if tmp_dest.stat().st_size < 100:
                log.warning("  [!] Downloaded file too small (%d B) -- skipping: %s",
                            tmp_dest.stat().st_size, url)
                tmp_dest.unlink()
                return False

            # Replace existing file (Path.rename fails on Windows if dest exists)
            if dest.exists():
                dest.unlink()
            tmp_dest.rename(dest)
            return True

    except requests.exceptions.HTTPError as exc:
        log.warning("  [!] HTTP %s -- skipping: %s", exc.response.status_code, url)
    except requests.exceptions.RequestException as exc:
        log.error("  [X] Network error downloading %s: %s", url, exc)
    finally:
        # Clean up partial downloads
        if tmp_dest.exists() and not dest.exists():
            tmp_dest.unlink(missing_ok=True)
    return False


# ---------------------------------------------
# MAIN SCRAPE CYCLE
# ---------------------------------------------

def scrape_once() -> dict:
    """
    Run a single scrape cycle:
    1. Fetch the resource list from the CKAN API.
    2. Download any *new* CSV files to data/raw_data/.
    3. Re-download the latest resource(s) if their server-side
       last_modified timestamp is newer than what we recorded.
    4. Return a summary dict.
    """
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    meta = _load_metadata()

    try:
        resources = _fetch_resource_list()
    except Exception as exc:
        log.error("Failed to fetch resource list: %s", exc)
        return {"error": str(exc)}

    stats = {"checked": 0, "downloaded": 0, "updated": 0,
             "skipped_exists": 0, "skipped_404": 0, "renamed": 0}

    # Identify the latest resource(s) by position (highest = most recent).
    # The current month's archive keeps getting updated on the server,
    # so we re-download it whenever last_modified changes.
    if resources:
        max_pos = max(r.get("position", 0) for r in resources)
    else:
        max_pos = -1

    # Track semantic filenames to handle duplicates (e.g. two "August 2023")
    seen_semantic: dict[str, int] = {}

    for res in resources:
        url = res.get("url", "")
        name = res.get("name", "unknown")
        resource_id = res.get("id", "")
        server_modified = res.get("last_modified") or res.get("metadata_modified") or ""
        position = res.get("position", -1)

        if not url:
            continue

        url_filename = _filename_from_url(url)

        # Derive a semantic filename from the resource name
        semantic = _semantic_filename(name)
        if semantic:
            # Handle duplicate resource names by appending _2, _3, etc.
            count = seen_semantic.get(semantic, 0) + 1
            seen_semantic[semantic] = count
            if count > 1:
                semantic = semantic.replace(".csv", f"_{count}.csv")
            filename = semantic
        elif url_filename and url_filename.endswith(".csv"):
            filename = url_filename
        else:
            filename = f"{resource_id}.csv"

        stats["checked"] += 1
        dest = RAW_DATA_DIR / filename

        # --- Rename: if the URL filename (tmp*) already exists on disk,
        #     rename it to the semantic name before anything else. ---
        if url_filename != filename:
            old_path = RAW_DATA_DIR / url_filename
            if old_path.exists() and not dest.exists():
                old_path.rename(dest)
                stats["renamed"] += 1
                log.info("  [mv] %s -> %s", url_filename, filename)

        # --- Is this one of the latest resources? ---
        is_latest = (position >= max_pos - 1)  # last 2 resources

        if dest.exists():
            if is_latest and meta.get(resource_id) != server_modified:
                # Server version was updated since our last download
                log.info("  [~] Update detected for [%s] (server: %s, local: %s)",
                         name, server_modified, meta.get(resource_id, "n/a"))
                log.info("  >> Re-downloading [%s] -> %s", name, filename)
                ok = _download_resource(url, dest)
                if ok:
                    stats["updated"] += 1
                    meta[resource_id] = server_modified
                    log.info("  [OK] Updated %s (%.1f KB)",
                             filename, dest.stat().st_size / 1024)
                else:
                    stats["skipped_404"] += 1
            else:
                stats["skipped_exists"] += 1
            continue

        # --- New file ---
        log.info("  >> Downloading [%s] -> %s", name, filename)
        ok = _download_resource(url, dest)
        if ok:
            stats["downloaded"] += 1
            meta[resource_id] = server_modified
            log.info("  [OK] Saved %s (%.1f KB)", filename, dest.stat().st_size / 1024)
        else:
            stats["skipped_404"] += 1

    _save_metadata(meta)

    log.info(
        "Cycle complete: checked=%d  new=%d  updated=%d  "
        "renamed=%d  already_exists=%d  404/skipped=%d",
        stats["checked"], stats["downloaded"], stats["updated"],
        stats["renamed"], stats["skipped_exists"], stats["skipped_404"],
    )
    return stats


def run_forever():
    """Scrape in a loop every SCRAPE_INTERVAL_SECONDS."""
    log.info("=" * 60)
    log.info("Nairobi AQ Scraper started - interval: %d min",
             SCRAPE_INTERVAL_SECONDS // 60)
    log.info("Download directory: %s", RAW_DATA_DIR)
    log.info("Metadata file:     %s", METADATA_PATH)
    log.info("=" * 60)

    while True:
        cycle_start = datetime.now(timezone.utc)
        log.info("--- Scrape cycle at %s ---", cycle_start.isoformat())

        scrape_once()

        elapsed = (datetime.now(timezone.utc) - cycle_start).total_seconds()
        sleep_for = max(0, SCRAPE_INTERVAL_SECONDS - elapsed)
        log.info("Sleeping %.0f s until next cycle ...\n", sleep_for)
        time.sleep(sleep_for)


# ---------------------------------------------
# CLI
# ---------------------------------------------

def rename_existing() -> dict:
    """
    Query the CKAN API and rename any tmp* files on disk
    to their semantic names, without downloading anything.
    """
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    try:
        resources = _fetch_resource_list()
    except Exception as exc:
        log.error("Failed to fetch resource list: %s", exc)
        return {"error": str(exc)}

    renamed = 0
    seen_semantic: dict[str, int] = {}

    for res in resources:
        url = res.get("url", "")
        name = res.get("name", "unknown")
        if not url:
            continue

        url_filename = _filename_from_url(url)
        semantic = _semantic_filename(name)
        if not semantic:
            continue

        count = seen_semantic.get(semantic, 0) + 1
        seen_semantic[semantic] = count
        if count > 1:
            semantic = semantic.replace(".csv", f"_{count}.csv")

        if url_filename == semantic:
            continue  # already has a good name

        old_path = RAW_DATA_DIR / url_filename
        new_path = RAW_DATA_DIR / semantic
        if old_path.exists() and not new_path.exists():
            old_path.rename(new_path)
            renamed += 1
            log.info("  [mv] %s -> %s", url_filename, semantic)

    log.info("Rename complete: %d files renamed.", renamed)
    return {"renamed": renamed}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Nairobi AQ data scraper")
    parser.add_argument(
        "--once", action="store_true",
        help="Run a single scrape cycle then exit (don't loop).",
    )
    parser.add_argument(
        "--rename", action="store_true",
        help="Only rename existing tmp* files to semantic names (no download).",
    )
    args = parser.parse_args()

    if args.rename:
        rename_existing()
    elif args.once:
        scrape_once()
    else:
        run_forever()
