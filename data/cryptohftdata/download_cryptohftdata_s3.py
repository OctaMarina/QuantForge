#!/usr/bin/env python3
"""Download selected historical CryptoHFTData raw files from temporary S3 access.

Example:
  python download_cryptohftdata_s3.py \
    --start 2025-10-01 --end 2026-09-30 \
    --exchanges binance_futures \
    --symbols BTCUSDT ETHUSDT SOLUSDT \
    --types liquidations open_interest mark_price \
    --output ./crypto_hft_raw

Required environment variables (generate temporary credentials in vendor dashboard):
  AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_SESSION_TOKEN,
  CHD_ENDPOINT, optionally CHD_BUCKET (defaults to cryptohftdata-processed).

This script only downloads publicly archived files. No features are derived, and
missing liquidation files must not automatically be interpreted as zero volume.

dotenv -f .env run -- python -u download_cryptohftdata_s3.py \
  --start 2025-06-28 \
  --end 2026-10-31 \
  --exchanges binance_futures \
  --symbols BTCUSDT \
  --types orderbook trades ticker liquidations open_interest mark_price \
  --output ./cryptohftdata_raw
"""

import argparse
import csv
from datetime import date, timedelta, datetime, timezone
import os
from pathlib import Path
import sys

import boto3
from botocore.config import Config

DATA_TYPES = {"liquidations", "open_interest", "mark_price", "trades", "orderbook", "ticker"}


def args_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--start", type=date.fromisoformat, default=date(2025, 10, 1))
    p.add_argument("--end", type=date.fromisoformat, default=date(2026, 9, 30))
    p.add_argument("--exchanges", nargs="+", default=["binance_futures"])
    p.add_argument("--symbols", nargs="+", default=["BTCUSDT", "ETHUSDT", "SOLUSDT"])
    p.add_argument("--types", nargs="+", default=["liquidations", "open_interest", "mark_price"])
    p.add_argument("--output", type=Path, default=Path("crypto_hft_raw"))
    p.add_argument("--dry-run", action="store_true", help="List selections without downloading")
    return p


def main():
    args = args_parser().parse_args()
    if args.start > args.end:
        raise SystemExit("Start date must be <= end date")
    if unknown := set(args.types) - DATA_TYPES:
        raise SystemExit(f"Unknown data types: {', '.join(sorted(unknown))}")
    for env in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN", "CHD_ENDPOINT"):
        if not os.environ.get(env):
            raise SystemExit(f"Missing {env}. Generate temporary S3 credentials in the dashboard.")

    session = boto3.session.Session(
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
        aws_session_token=os.environ["AWS_SESSION_TOKEN"],
        region_name="auto",
    )
    s3 = session.client(
        "s3", endpoint_url=os.environ["CHD_ENDPOINT"],
        config=Config(retries={"max_attempts": 8, "mode": "standard"}),
    )
    bucket = os.environ.get("CHD_BUCKET", "cryptohftdata-processed")
    args.output.mkdir(parents=True, exist_ok=True)
    log_path = args.output / "download_manifest.csv"
    existing = log_path.exists()
    suffixes = {f"{s}_{t}.parquet{ext}" for s in args.symbols for t in args.types for ext in ("", ".zst")}
    metrics = {"selected": 0, "downloaded": 0, "skipped": 0, "errors": 0, "bytes": 0}

    with log_path.open("a", newline="", encoding="utf8") as logfile:
        writer = csv.writer(logfile)
        if not existing:
            writer.writerow(["logged_at_utc", "source_key", "status", "size_bytes", "message"])
        current = args.start
        while current <= args.end:
            for exchange in args.exchanges:
                prefix = f"{exchange}/{current.isoformat()}/"
                print(f"Listing {prefix}", flush=True)
                try:
                    pages = s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix)
                    for page in pages:
                        for item in page.get("Contents", []):
                            key = item["Key"]
                            if key.rsplit("/", 1)[-1] not in suffixes:
                                continue
                            metrics["selected"] += 1
                            dest = args.output / key
                            expected = int(item["Size"])
                            if dest.exists() and dest.stat().st_size == expected:
                                metrics["skipped"] += 1
                                continue
                            if args.dry_run:
                                print("WOULD DOWNLOAD", key, expected, "bytes")
                                continue
                            dest.parent.mkdir(parents=True, exist_ok=True)
                            tmp = dest.with_suffix(dest.suffix + ".part")
                            try:
                                s3.download_file(bucket, key, str(tmp))
                                if tmp.stat().st_size != expected:
                                    raise IOError(f"Size mismatch: {tmp.stat().st_size} vs {expected}")
                                tmp.replace(dest)
                                metrics["downloaded"] += 1
                                metrics["bytes"] += expected
                                status, message = "ok", ""
                            except Exception as e:
                                tmp.unlink(missing_ok=True)
                                metrics["errors"] += 1
                                status, message = "error", str(e)
                                print("FAILED:", key, message, file=sys.stderr)
                            writer.writerow([datetime.now(timezone.utc).isoformat(), key, status, expected, message])
                            logfile.flush()
                except Exception as e:
                    metrics["errors"] += 1
                    writer.writerow([datetime.now(timezone.utc).isoformat(), prefix, "list_error", "", str(e)])
                    logfile.flush()
                    print(f"Cannot list {prefix}: {e}", file=sys.stderr)
                    print("Credentials may have expired. Refresh credentials and rerun to resume.", file=sys.stderr)
                    print(metrics)
                    return 2
            current += timedelta(days=1)
    print("Done:", metrics)
    print("Manifest:", log_path)
    print("Note: no-file != verified zero liquidations. Audit source coverage separately.")
    print("For files ending .parquet.zst, use: zstd -d -k <filename>.parquet.zst")
    return 1 if metrics["errors"] else 0


if __name__ == "__main__":
    sys.exit(main())
