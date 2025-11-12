# scripts/fetch_open_data.py

"""
Fetch ECMWF IFS HRES open data subset for Fennoscandia.

This script is designed to be conservative and robust on GitHub Actions:
- It tries to download minimal GRIB2 files needed for the selected variables/levels.
- If a file is missing or download fails, the pipeline will still continue (the renderer will create placeholders).

IMPORTANT:
- ECMWF publishes IFS Open Data via public endpoints. Exact mirror URLs and directory
  layout can vary; set BASE_URL below to a valid mirror you have tested.
- Start with a small set of fields; expand when stable.
"""

import os, datetime as dt, pathlib, sys, time
from typing import Optional
import requests

# -------- CONFIG --------
# Fennoscandia bbox (lon/lat), used in renderer (we fetch global/slice later when possible)
BBOX = dict(west=5.0, south=55.0, east=35.5, north=72.5)

# Variables
SFC_PARAMS = ["msl", "2t", "tp", "tcwv", "ssr", "str"]
PL_PARAMS  = ["u", "v", "gh"]
PL_LEVELS  = [850, 500]  # hPa

# Steps to use (hours). Keep light initially.
STEPS = [0, 6, 12, 24]

# Where to store raw files
OUTDIR = pathlib.Path("data_raw"); OUTDIR.mkdir(parents=True, exist_ok=True)

# ---- SELECT MIRROR/BASE URL ----
# Please edit BASE_URL to a working ECMWF open data mirror with HRES GRIB2
# For example (this is a placeholder; replace with the mirror you use):
BASE_URL = os.environ.get("ECMWF_OPEN_BASE_URL", "").rstrip("/")  # allow override via env/secret
if not BASE_URL:
    print("[INFO] No BASE_URL set via ECMWF_OPEN_BASE_URL; downloads will be skipped (placeholders used).")

def pick_run(now_utc: dt.datetime) -> dt.datetime:
    # Pick the latest 6-hourly cycle
    hour = (now_utc.hour // 6) * 6
    return now_utc.replace(hour=hour, minute=0, second=0, microsecond=0)

def ensure_download(url: str, local_path: pathlib.Path, retries: int = 2) -> bool:
    try:
        local_path.parent.mkdir(parents=True, exist_ok=True)
        for attempt in range(retries+1):
            try:
                with requests.get(url, stream=True, timeout=60) as r:
                    if r.status_code != 200:
                        print(f"[WARN] HTTP {r.status_code} for {url}")
                        time.sleep(2)
                        continue
                    with open(local_path, "wb") as f:
                        for chunk in r.iter_content(chunk_size=1<<20):
                            if chunk:
                                f.write(chunk)
                size = local_path.stat().st_size
                if size < 1000:
                    print(f"[WARN] File too small ({size} B): {local_path.name}")
                    return False
                print(f"[OK] Downloaded {local_path.name} ({size/1e6:.2f} MB)")
                return True
            except requests.RequestException as e:
                print(f"[WARN] attempt {attempt+1} failed for {url}: {e}")
                time.sleep(2)
        return False
    except Exception as e:
        print(f"[ERROR] ensure_download: {e}")
        return False

def build_filename(param: str, leveltype: str, level: Optional[int], run: dt.datetime) -> str:
    runstr = run.strftime("%Y%m%d%H")
    name = f"{param}_{leveltype}"
    if level is not None:
        name += f"_{level}"
    name += f"_{runstr}.grib2"
    return name

def main():
    now = dt.datetime.utcnow()
    run = pick_run(now)
    runstr = run.strftime("%Y%m%d%H")
    print("[INFO] Selected IFS run:", runstr)

    # NOTE:
    # You must adapt URL templates to your chosen ECMWF open data mirror.
    # Many mirrors expose GRIB files by date/run and parameter; e.g.:
    #   f"{BASE_URL}/hres/{YYYYMMDD}/{HH}/.../msl_sfc_{YYYYMMDDHH}.grib2"
    #
    # In this template, we try a generic path if BASE_URL is set.
    downloaded_any = False
    for p in SFC_PARAMS:
        fname = build_filename(p, "sfc", None, run)
        local = OUTDIR / fname
        if BASE_URL:
            # Example (you MUST update to real pattern of your mirror)
            url = f"{BASE_URL}/hres/{run:%Y%m%d}/{run:%H}/{fname}"
            ok = ensure_download(url, local)
            downloaded_any |= ok
        else:
            print(f"[SKIP] {fname} (no BASE_URL)")

    for p in PL_PARAMS:
        for lvl in PL_LEVELS:
            fname = build_filename(p, "pl", lvl, run)
            local = OUTDIR / fname
            if BASE_URL:
                url = f"{BASE_URL}/hres/{run:%Y%m%d}/{run:%H}/{fname}"
                ok = ensure_download(url, local)
                downloaded_any |= ok
            else:
                print(f"[SKIP] {fname} (no BASE_URL)")

    if not downloaded_any:
        print("[INFO] No downloads performed. The renderer will produce placeholder overlays so the site still loads.")

if __name__ == "__main__":
    main()
