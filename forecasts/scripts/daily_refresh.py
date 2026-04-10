"""
daily_refresh.py
-----------------
Single entry point for the daily forecast update.
Run this every morning — takes ~2 minutes, no tile rebuild needed.

Steps:
  1. Fetch latest NAC forecasts
  2. Generate forecast.json (the only file that changes daily)
  3. Upload to Cloudflare R2 (if configured)
  4. Log summary

Usage:
    python daily_refresh.py                    # fetch + save locally
    python daily_refresh.py --upload           # fetch + upload to R2
    python daily_refresh.py --dry              # fetch only, no save

Schedule (Windows Task Scheduler):
    Action: python C:\...\daily_refresh.py --upload
    Trigger: Daily at 07:00 AM

Schedule (GitHub Actions / cron):
    See .github/workflows/daily_refresh.yml
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent))


def run(upload: bool = False, dry: bool = False) -> None:
    start = time.time()
    log.info("=== Daily forecast refresh — %s ===",
             datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"))

    # Step 1: Build forecast.json
    log.info("Step 1/2: Building forecast.json...")
    from build_forecast_json import build_forecast_json
    result = build_forecast_json(dry_run=dry)
    zones_loaded = len(result.get("zones", {}))
    log.info("  %d zones loaded for date %s", zones_loaded, result.get("date"))

    # Step 2: Upload to R2 (optional)
    if upload and not dry:
        log.info("Step 2/2: Uploading to Cloudflare R2...")
        upload_to_r2()
    else:
        log.info("Step 2/2: Skipping upload (local only)")

    elapsed = time.time() - start
    log.info("=== Refresh complete in %.1f seconds ===", elapsed)


def upload_to_r2() -> None:
    """
    Upload forecast.json to Cloudflare R2.

    Requires environment variables:
      R2_ACCOUNT_ID   — Cloudflare account ID
      R2_ACCESS_KEY   — R2 access key
      R2_SECRET_KEY   — R2 secret key
      R2_BUCKET       — bucket name (e.g. 'avi-mapping')
      R2_PUBLIC_URL   — public URL base (e.g. 'https://tiles.yourdomain.com')
    """
    try:
        import boto3
        from botocore.config import Config
    except ImportError:
        log.error("boto3 not installed — run: pip install boto3")
        return

    account_id  = os.environ.get("R2_ACCOUNT_ID")
    access_key  = os.environ.get("R2_ACCESS_KEY")
    secret_key  = os.environ.get("R2_SECRET_KEY")
    bucket      = os.environ.get("R2_BUCKET", "avi-mapping")

    if not all([account_id, access_key, secret_key]):
        log.warning("R2 credentials not set — skipping upload")
        log.warning("  Set R2_ACCOUNT_ID, R2_ACCESS_KEY, R2_SECRET_KEY env vars")
        return

    endpoint = f"https://{account_id}.r2.cloudflarestorage.com"
    s3 = boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        config=Config(signature_version="s3v4"),
        region_name="auto",
    )

    data_dir = Path(__file__).parent.parent.parent / "web" / "data"
    forecast_path = data_dir / "forecast.json"

    if not forecast_path.exists():
        log.error("forecast.json not found")
        return

    log.info("  Uploading forecast.json...")
    s3.upload_file(
        str(forecast_path),
        bucket,
        "data/forecast.json",
        ExtraArgs={
            "ContentType": "application/json",
            "CacheControl": "public, max-age=3600",  # 1 hour cache
        },
    )
    log.info("  Uploaded forecast.json to R2 bucket '%s'", bucket)


def upload_tiles_to_r2(tile_type: str = "identity") -> None:
    """
    One-time upload of static tiles to R2.
    Run this once after build_identity_tiles.py completes.

    tile_type: 'identity' for identity tiles
    """
    try:
        import boto3
        from botocore.config import Config
    except ImportError:
        log.error("boto3 not installed — run: pip install boto3")
        return

    account_id = os.environ.get("R2_ACCOUNT_ID")
    access_key = os.environ.get("R2_ACCESS_KEY")
    secret_key = os.environ.get("R2_SECRET_KEY")
    bucket     = os.environ.get("R2_BUCKET", "avi-mapping")

    if not all([account_id, access_key, secret_key]):
        log.warning("R2 credentials not set")
        return

    endpoint = f"https://{account_id}.r2.cloudflarestorage.com"
    s3 = boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        config=Config(signature_version="s3v4"),
        region_name="auto",
    )

    tiles_dir = Path(__file__).parent.parent.parent / "web" / "tiles" / tile_type
    if not tiles_dir.exists():
        log.error("Tiles directory not found: %s", tiles_dir)
        return

    all_tiles = list(tiles_dir.rglob("*.png"))
    log.info("Uploading %d %s tiles to R2...", len(all_tiles), tile_type)

    uploaded = 0
    for i, tile_path in enumerate(all_tiles):
        rel = tile_path.relative_to(tiles_dir.parent.parent)  # web/tiles/identity/z/x/y.png
        key = str(rel).replace("\\", "/")

        s3.upload_file(
            str(tile_path),
            bucket,
            key,
            ExtraArgs={
                "ContentType": "image/png",
                "CacheControl": "public, max-age=31536000",  # 1 year (static)
            },
        )
        uploaded += 1
        if uploaded % 1000 == 0:
            log.info("  %d / %d tiles uploaded...", uploaded, len(all_tiles))

    log.info("Done — %d tiles uploaded to R2", uploaded)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Daily forecast refresh")
    parser.add_argument("--upload",       action="store_true",
                        help="Upload forecast.json to Cloudflare R2")
    parser.add_argument("--upload-tiles", type=str, metavar="TYPE",
                        help="One-time tile upload to R2 (e.g. --upload-tiles identity)")
    parser.add_argument("--dry",          action="store_true",
                        help="Fetch only, no save or upload")
    args = parser.parse_args()

    if args.upload_tiles:
        upload_tiles_to_r2(args.upload_tiles)
    else:
        run(upload=args.upload, dry=args.dry)
