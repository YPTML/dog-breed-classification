import csv
import hashlib
import os
import re
import time
from pathlib import Path
from urllib.parse import urlparse

import requests


# ============================================================
# CONFIG
# ============================================================

ROOT = Path(r"E:\ML\ML PBC(3.9.26)")

OUT_DIR = ROOT / "data" / "new_breeds_v3" / "phu_quoc_ridgeback"
META_DIR = ROOT / "metadata"

OUT_DIR.mkdir(parents=True, exist_ok=True)
META_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH = META_DIR / "phu_quoc_ridgeback_wikimedia.csv"

API_URL = "https://commons.wikimedia.org/w/api.php"
CATEGORY = "Category:Phu Quoc ridgeback dog"

# Tải thumbnail tối đa ~1280px là đủ cho ML
THUMB_WIDTH = 1280

# Nghỉ giữa mỗi file để tránh Wikimedia rate-limit
DOWNLOAD_DELAY_SECONDS = 3

MAX_RETRIES = 8

ALLOWED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
}

session = requests.Session()

session.headers.update({
    "User-Agent": (
        "ML-PBC-Dataset-Builder/1.1 "
        "(educational machine-learning dataset curation)"
    ),
    "Accept": "*/*",
})


# ============================================================
# HELPERS
# ============================================================

def clean_text(text):
    if not text:
        return ""

    text = re.sub(r"<[^>]+>", "", text)
    text = text.replace("&nbsp;", " ")
    text = text.replace("&amp;", "&")
    text = text.replace("&quot;", '"')

    return text.strip()


def sha256_file(path):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)

            if not chunk:
                break

            h.update(chunk)

    return h.hexdigest()


def wait_for_retry(response, attempt):
    retry_after = response.headers.get("Retry-After")

    if retry_after:
        try:
            seconds = int(retry_after)
        except ValueError:
            seconds = 10
    else:
        # 5, 10, 20, 40, 60...
        seconds = min(5 * (2 ** attempt), 60)

    print(f"  RATE LIMITED. Waiting {seconds}s...")
    time.sleep(seconds)


def api_get(params):
    for attempt in range(MAX_RETRIES):

        response = session.get(
            API_URL,
            params=params,
            timeout=60,
        )

        if response.status_code == 429:
            wait_for_retry(response, attempt)
            continue

        if response.status_code >= 500:
            seconds = min(5 * (2 ** attempt), 60)

            print(
                f"  Wikimedia server error "
                f"{response.status_code}. Waiting {seconds}s..."
            )

            time.sleep(seconds)
            continue

        response.raise_for_status()

        return response.json()

    raise RuntimeError(
        "Wikimedia API remained rate-limited after all retries."
    )


# ============================================================
# STEP 1 - CATEGORY FILE LIST
# ============================================================

def get_category_files():
    files = []
    cmcontinue = None

    while True:

        params = {
            "action": "query",
            "format": "json",
            "list": "categorymembers",
            "cmtitle": CATEGORY,
            "cmtype": "file",
            "cmlimit": "500",
        }

        if cmcontinue:
            params["cmcontinue"] = cmcontinue

        data = api_get(params)

        files.extend(
            data["query"]["categorymembers"]
        )

        if "continue" not in data:
            break

        cmcontinue = data["continue"]["cmcontinue"]

        time.sleep(2)

    return files


# ============================================================
# STEP 2 - GET METADATA IN BATCHES
# ============================================================

def chunks(items, size):
    for i in range(0, len(items), size):
        yield items[i:i + size]


def get_metadata(files):
    title_map = {}

    titles = [
        item["title"]
        for item in files
        if not item["title"].lower().endswith(".svg")
    ]

    # Chỉ 25 title / request -> rất nhẹ
    for batch_index, batch in enumerate(
        chunks(titles, 25),
        start=1
    ):

        print(
            f"Metadata batch {batch_index}: "
            f"{len(batch)} files"
        )

        params = {
            "action": "query",
            "format": "json",
            "prop": "imageinfo",
            "titles": "|".join(batch),
            "iiprop": "url|size|mime|extmetadata",
            "iiurlwidth": str(THUMB_WIDTH),
        }

        data = api_get(params)

        pages = data["query"]["pages"]

        for page in pages.values():

            title = page.get("title")

            infos = page.get("imageinfo", [])

            if not infos:
                continue

            info = infos[0]

            metadata = info.get(
                "extmetadata",
                {}
            )

            def meta(key):
                item = metadata.get(key, {})
                return clean_text(
                    item.get("value", "")
                )

            title_map[title] = {
                # ưu tiên thumbnail ~1280px
                "download_url":
                    info.get("thumburl")
                    or info.get("url"),

                "original_url":
                    info.get("url"),

                "description_url":
                    info.get("descriptionurl"),

                "width":
                    info.get("thumbwidth")
                    or info.get("width"),

                "height":
                    info.get("thumbheight")
                    or info.get("height"),

                "original_width":
                    info.get("width"),

                "original_height":
                    info.get("height"),

                "mime":
                    info.get("mime"),

                "artist":
                    meta("Artist"),

                "license":
                    meta("LicenseShortName"),

                "license_url":
                    meta("LicenseUrl"),

                "credit":
                    meta("Credit"),

                "description":
                    meta("ImageDescription"),

                "date_time":
                    meta("DateTimeOriginal")
                    or meta("DateTime"),

                "source":
                    meta("Source"),
            }

        # tránh burst API
        time.sleep(3)

    return title_map


# ============================================================
# DOWNLOAD WITH RETRIES
# ============================================================

def download_file(url, destination):

    for attempt in range(MAX_RETRIES):

        response = session.get(
            url,
            stream=True,
            timeout=120,
        )

        if response.status_code == 429:
            wait_for_retry(response, attempt)
            continue

        if response.status_code >= 500:

            seconds = min(
                5 * (2 ** attempt),
                60
            )

            print(
                f"  Server error "
                f"{response.status_code}. "
                f"Waiting {seconds}s..."
            )

            time.sleep(seconds)
            continue

        response.raise_for_status()

        temp_path = destination.with_suffix(
            destination.suffix + ".part"
        )

        with open(temp_path, "wb") as f:

            for chunk in response.iter_content(
                chunk_size=1024 * 1024
            ):

                if chunk:
                    f.write(chunk)

        if temp_path.stat().st_size == 0:
            temp_path.unlink(
                missing_ok=True
            )
            raise RuntimeError(
                "Downloaded file is empty."
            )

        temp_path.replace(destination)

        return

    raise RuntimeError(
        "Download failed after all retries."
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 72)
    print(
        "PHU QUOC RIDGEBACK - "
        "WIKIMEDIA COMMONS DOWNLOADER V2"
    )
    print("=" * 72)

    print(f"Category : {CATEGORY}")
    print(f"Output   : {OUT_DIR}")
    print(f"Metadata : {CSV_PATH}")

    print("\n[1/4] Reading category...")

    files = get_category_files()

    print(
        f"Found {len(files)} Wikimedia entries."
    )

    print("\n[2/4] Downloading metadata in batches...")

    metadata_map = get_metadata(files)

    print(
        f"Metadata available for "
        f"{len(metadata_map)} raster files."
    )

    records = []

    downloaded = 0
    existing = 0
    skipped = 0
    failed = 0

    raster_index = 0

    for original_index, item in enumerate(
        files,
        start=1
    ):

        title = item["title"]

        print(
            f"\n[{original_index}/{len(files)}] "
            f"{title}"
        )

        if title.lower().endswith(".svg"):

            print("  SKIP: SVG not suitable for breed training.")
            skipped += 1
            continue

        info = metadata_map.get(title)

        if not info:

            print("  SKIP: no metadata.")
            skipped += 1
            continue

        url = info["download_url"]

        if not url:

            print("  SKIP: no download URL.")
            skipped += 1
            continue

        ext = Path(
            urlparse(url).path
        ).suffix.lower()

        if ext not in ALLOWED_EXTENSIONS:

            # Thumbnail URLs sometimes have odd suffixes;
            # fall back to JPG when MIME is JPEG.
            mime = info.get("mime", "")

            if mime == "image/jpeg":
                ext = ".jpg"
            elif mime == "image/png":
                ext = ".png"
            else:
                print(
                    f"  SKIP: unsupported "
                    f"extension/mime: {ext} / {mime}"
                )
                skipped += 1
                continue

        raster_index += 1

        filename = (
            f"phu_quoc_ridgeback_"
            f"{raster_index:04d}{ext}"
        )

        destination = (
            OUT_DIR / filename
        )

        try:

            if (
                destination.exists()
                and destination.stat().st_size > 0
            ):

                print(
                    f"  EXISTS: {filename}"
                )

                existing += 1

            else:

                print(
                    f"  Downloading -> "
                    f"{filename}"
                )

                download_file(
                    url,
                    destination
                )

                downloaded += 1

                print(
                    f"  OK: "
                    f"{destination.stat().st_size:,} bytes"
                )

            sha256 = sha256_file(
                destination
            )

            records.append({
                "local_file":
                    str(
                        destination.relative_to(ROOT)
                    ).replace("\\", "/"),

                "wikimedia_title":
                    title,

                "source_url":
                    info["description_url"],

                "download_url":
                    url,

                "original_url":
                    info["original_url"],

                "width":
                    info["width"],

                "height":
                    info["height"],

                "original_width":
                    info["original_width"],

                "original_height":
                    info["original_height"],

                "mime":
                    info["mime"],

                "artist":
                    info["artist"],

                "license":
                    info["license"],

                "license_url":
                    info["license_url"],

                "credit":
                    info["credit"],

                "description":
                    info["description"],

                "date_time":
                    info["date_time"],

                "source":
                    info["source"],

                "sha256":
                    sha256,
            })

        except Exception as e:

            failed += 1

            print(
                f"  ERROR: "
                f"{type(e).__name__}: {e}"
            )

        # rất quan trọng
        time.sleep(
            DOWNLOAD_DELAY_SECONDS
        )

    print("\n[3/4] Writing metadata CSV...")

    fieldnames = [
        "local_file",
        "wikimedia_title",
        "source_url",
        "download_url",
        "original_url",
        "width",
        "height",
        "original_width",
        "original_height",
        "mime",
        "artist",
        "license",
        "license_url",
        "credit",
        "description",
        "date_time",
        "source",
        "sha256",
    ]

    with open(
        CSV_PATH,
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(records)

    print("\n[4/4] COMPLETE")

    print("\nSummary")
    print("-" * 40)
    print(
        f"Wikimedia entries : {len(files)}"
    )
    print(
        f"Metadata records   : {len(metadata_map)}"
    )
    print(
        f"Downloaded         : {downloaded}"
    )
    print(
        f"Already existed    : {existing}"
    )
    print(
        f"Successful records : {len(records)}"
    )
    print(
        f"Skipped            : {skipped}"
    )
    print(
        f"Failed             : {failed}"
    )

    print("\nImages:")
    print(OUT_DIR)

    print("\nMetadata:")
    print(CSV_PATH)


if __name__ == "__main__":
    main()