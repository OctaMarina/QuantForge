#!/usr/bin/env python3
"""Download hourly CHD archives; auto-refresh JWT; reuse and validate S3 files.

Requires Python 3.10+. Reads .env itself. No AWS credentials are used.
Validation checks readability, NOT market-data completeness or source equality.
"""
import argparse
import base64
import csv
from datetime import date, datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
import json
import os
from pathlib import Path
import re
import shutil
import sqlite3
import tempfile
import time

import pyarrow.parquet as pq
import requests
import zstandard as zstd
from dotenv import dotenv_values

API = 'https://api.cryptohftdata.com/v1'
TYPES = ('orderbook', 'trades', 'ticker', 'liquidations', 'open_interest', 'mark_price')
ZSTD_MAGIC = b'\x28\xb5\x2f\xfd'


class AuthError(RuntimeError):
    pass


def log(message):
    print(f'{datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S} UTC | {message}', flush=True)


def retry_delay(response, fallback):
    value = response.headers.get('Retry-After', '')
    try:
        return max(0.0, float(value))
    except ValueError:
        try:
            return max(0.0, parsedate_to_datetime(value).timestamp() - time.time())
        except (ValueError, TypeError, OverflowError):
            return fallback


def pause(seconds):
    # Long server-directed waits remain visible and interruptible.
    deadline = time.monotonic() + seconds
    while (remaining := deadline - time.monotonic()) > 0:
        if remaining > 60:
            log(f'Waiting: {remaining:.0f}s remaining')
        time.sleep(min(30, remaining))


def parquet_rows(path):
    """Decode every column in bounded row batches, including compressed archives."""
    path = Path(path)
    with path.open('rb') as source:
        magic = source.read(4)
        source.seek(0)
        if magic == ZSTD_MAGIC:
            # One decompressed hourly file at a time; released on every exit path.
            with tempfile.TemporaryDirectory(prefix='.chd_validate_', dir=path.parent) as tmp:
                plain = Path(tmp) / 'data.parquet'
                with zstd.ZstdDecompressor().stream_reader(source) as reader, plain.open('wb') as out:
                    shutil.copyfileobj(reader, out, 1024 * 1024)
                return parquet_rows(plain)
        if magic != b'PAR1':
            raise ValueError('Not a Parquet or Zstandard archive')
    try:
        with pq.ParquetFile(path) as parquet:
            rows = sum(batch.num_rows for batch in parquet.iter_batches(batch_size=16384, use_threads=False))
            if rows != parquet.metadata.num_rows:
                raise ValueError('Parquet row count mismatch')
    except OSError as exc:
        if exc.errno is not None:
            raise
        # Arrow uses OSError without errno for unreadable/corrupt pages.
        raise ValueError('Parquet decoding failed') from None
    return rows


class Client:
    def __init__(self, key, retries=6, interval=1.1):
        self.key, self.retries, self.interval = key, retries, interval
        self.session = requests.Session()
        self.token, self.refresh_at, self.last_request = None, 0.0, 0.0

    def throttle(self):
        pause(max(0, self.last_request + self.interval - time.monotonic()))
        self.last_request = time.monotonic()

    def ensure_token(self):
        if self.token and time.time() < self.refresh_at:
            return
        for attempt in range(self.retries):
            self.throttle()
            try:
                with self.session.post(API + '/jwt-token', headers={
                    'X-API-Key': self.key, 'Content-Type': 'application/json'
                }, timeout=(15, 90), allow_redirects=False) as response:
                    if response.status_code in (401, 403):
                        raise AuthError('API key rejected. Check CRYPTOHFTDATA_API_KEY in .env.')
                    if response.status_code == 429 or response.status_code >= 500:
                        delay = retry_delay(response, min(60, 2 ** attempt))
                    elif response.status_code != 200:
                        raise AuthError(f'Token endpoint returned HTTP {response.status_code}.')
                    else:
                        data = response.json()
                        token = data.get('jwt_token')
                        if not isinstance(token, str) or not token:
                            raise AuthError('Token response has no jwt_token.')
                        now = time.time()
                        expiry = now + float(data.get('expires_in') or 14400)
                        # Read exp only for scheduling, never as an authentication check.
                        try:
                            part = token.split('.')[1]
                            claims = json.loads(base64.urlsafe_b64decode(part + '=' * (-len(part) % 4)))
                            expiry = min(expiry, float(claims['exp']))
                        except (ValueError, IndexError, KeyError, TypeError):
                            pass
                        if expiry <= now:
                            raise AuthError('Server returned an expired token; check system clock.')
                        self.token = token
                        self.refresh_at = now + max(1, (expiry - now) - min(300, (expiry - now) / 2))
                        log(f'Token renewed; next scheduled renewal in {(self.refresh_at-now)/60:.0f} min')
                        return
            except requests.RequestException:
                delay = min(60, 2 ** attempt)
            except (ValueError, TypeError):
                raise AuthError('Invalid token response from server.') from None
            log(f'Token request retry {attempt+1}/{self.retries}')
            pause(delay)
        raise AuthError('Could not obtain token after retries. Rerun later; local files are retained.')

    def download(self, key, output):
        """Return (path, rows), or None for 404. Commit only validated complete files."""
        base = output / key
        base.parent.mkdir(parents=True, exist_ok=True)
        part = base.with_name(base.name + '.rest.part')
        auth_retries = 0
        last_error = 'download failed'
        for attempt in range(self.retries):
            self.ensure_token()
            self.throttle()
            delay = min(60, 2 ** attempt)
            try:
                # requests removes Authorization on cross-host redirects.
                with self.session.get(API + '/download', params={'file': key}, headers={
                    'Authorization': 'Bearer ' + self.token, 'Accept-Encoding': 'identity'
                }, stream=True, timeout=(15, 120)) as response:
                    status = response.status_code
                    if status in (401, 403):
                        auth_retries += 1
                        if auth_retries > 1:
                            raise AuthError('Download still forbidden after token renewal; stopping.')
                        self.token = None
                        log('Authentication rejected; renewing token and retrying the same file')
                        continue
                    if status == 404:
                        return None
                    if status == 429 or status >= 500:
                        last_error = f'HTTP {status}'
                        delay = retry_delay(response, delay)
                    elif status != 200:
                        raise RuntimeError(f'Download returned HTTP {status}')
                    else:
                        length = response.headers.get('Content-Length')
                        count = 0
                        with part.open('wb') as stream:
                            for chunk in response.iter_content(1024 * 1024):
                                if chunk:
                                    stream.write(chunk)
                                    count += len(chunk)
                            stream.flush()
                            os.fsync(stream.fileno())
                        if length and not response.headers.get('Content-Encoding') and count != int(length):
                            raise ValueError('Content-Length mismatch')
                        rows = parquet_rows(part)
                        with part.open('rb') as stream:
                            compressed = stream.read(4) == ZSTD_MAGIC
                        dest = Path(str(base) + '.zst') if compressed else base
                        os.replace(part, dest)
                        return dest, rows
            except requests.RequestException as exc:
                # Never print request URLs/headers or raw exceptions that could expose signed URLs.
                last_error = type(exc).__name__
            except (ValueError, zstd.ZstdError) as exc:
                last_error = type(exc).__name__
            finally:
                part.unlink(missing_ok=True)
            log(f'RETRY {attempt+1}/{self.retries} {key}: {last_error}; waiting {delay:.0f}s')
            pause(delay)
        raise RuntimeError(f'Failed after {self.retries} attempts: {last_error}')


class Ledger:
    def __init__(self, output):
        self.output = output
        self.db = sqlite3.connect(output / 'rest_download_state.sqlite')
        self.db.execute('CREATE TABLE IF NOT EXISTS files (key TEXT PRIMARY KEY, status TEXT, path TEXT, size INTEGER, mtime INTEGER, rows INTEGER, checked TEXT, note TEXT)')

    def record(self, key, status, path=None, rows=None, note=''):
        stat = path.stat() if path else None
        self.db.execute('INSERT OR REPLACE INTO files VALUES (?,?,?,?,?,?,?,?)', (
            key, status, str(path.relative_to(self.output)) if path else None,
            stat.st_size if stat else None, stat.st_mtime_ns if stat else None, rows,
            datetime.now(timezone.utc).isoformat(), note))
        self.db.commit()

    def local(self, key, revalidate=False):
        cached = self.db.execute('SELECT status,path,size,mtime FROM files WHERE key=?', (key,)).fetchone()
        if cached and cached[0] == 'ok' and not revalidate:
            path = self.output / cached[1]
            if path.is_file():
                stat = path.stat()
                if (stat.st_size, stat.st_mtime_ns) == tuple(cached[2:]):
                    return 'cached'
        for path in (self.output / key, self.output / (key + '.zst')):
            if not path.is_file():
                continue
            log(f'VALIDATE local {path.relative_to(self.output)}')
            try:
                rows = parquet_rows(path)
            except OSError:
                # Disk/permission failures should stop, not trigger more downloads.
                raise
            except Exception as exc:
                log(f'INVALID local {path.name} ({type(exc).__name__}); will try to replace it')
                continue
            self.record(key, 'ok', path, rows, 'Local file decoded; not compared with server')
            return 'validated'
        return None

    def export(self):
        for filename, query in (
            ('rest_manifest.csv', 'SELECT * FROM files ORDER BY key'),
            ('rest_unavailable_or_errors.csv', "SELECT * FROM files WHERE status != 'ok' ORDER BY key"),
        ):
            with (self.output / filename).open('w', newline='', encoding='utf-8') as out:
                cursor = self.db.execute(query)
                writer = csv.writer(out)
                writer.writerow([column[0] for column in cursor.description])
                writer.writerows(cursor)


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--start', type=date.fromisoformat, required=True)
    parser.add_argument('--end', type=date.fromisoformat, required=True, help='Inclusive UTC date')
    parser.add_argument('--exchanges', nargs='+', default=['binance_futures'])
    parser.add_argument('--symbols', nargs='+', default=['BTCUSDT'])
    parser.add_argument('--types', nargs='+', choices=TYPES, default=list(TYPES))
    parser.add_argument('--output', type=Path, default=Path('cryptohftdata_raw'))
    parser.add_argument('--env-file', type=Path, default=Path('.env'))
    parser.add_argument('--revalidate', action='store_true', help='Decode previously validated local files again')
    parser.add_argument('--min-interval', type=float, default=1.1, help='Seconds between API requests')
    parser.add_argument('--retries', type=int, default=6)
    args = parser.parse_args()
    if args.start > args.end or args.min_interval < 0 or args.retries < 2:
        parser.error('Invalid date range, interval or retry count')
    if not all(re.fullmatch(r'[A-Za-z0-9_:@.-]+', value) and '..' not in value for value in args.exchanges + args.symbols):
        parser.error('Invalid exchange or symbol')
    return args


def main():
    args = arguments()
    # Explicit .env wins over stale parent-shell values. Never rewrite the user's .env.
    settings = dotenv_values(args.env_file)
    api_key = settings.get('CRYPTOHFTDATA_API_KEY') or os.environ.get('CRYPTOHFTDATA_API_KEY')
    if not api_key:
        raise SystemExit('Add CRYPTOHFTDATA_API_KEY to .env (the ordinary API key, not S3 credentials).')
    args.output.mkdir(parents=True, exist_ok=True)
    # Avoid two downloader instances using the same .part files and output state.
    import fcntl
    lock = (args.output / '.rest_download.lock').open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit('Another REST downloader is using this output folder.')
    ledger = Ledger(args.output)
    client = Client(api_key, args.retries, args.min_interval)
    counts = dict(downloaded=0, validated=0, cached=0, unavailable=0, errors=0)
    exit_code = 0
    current_key = None
    try:
        client.ensure_token()
        current = args.start
        while current <= args.end:
            log(f'DAY {current} | {counts}')
            for exchange in args.exchanges:
                for hour in range(24):
                    for symbol in args.symbols:
                        for kind in args.types:
                            current_key = f'{exchange}/{current}/{hour:02d}/{symbol}_{kind}.parquet'
                            state = ledger.local(current_key, args.revalidate)
                            if state:
                                counts[state] += 1
                                continue
                            try:
                                result = client.download(current_key, args.output)
                            except AuthError:
                                raise
                            except RuntimeError as exc:
                                counts['errors'] += 1
                                ledger.record(current_key, 'error', note=str(exc))
                                log(f'ERROR {current_key}: {exc}')
                                exit_code = 1
                                continue
                            if result is None:
                                counts['unavailable'] += 1
                                ledger.record(current_key, 'unavailable', note='HTTP 404; not proof of zero activity')
                                log(f'UNAVAILABLE {current_key}')
                            else:
                                path, rows = result
                                counts['downloaded'] += 1
                                ledger.record(current_key, 'ok', path, rows, 'Downloaded and decoded')
                                log(f'DOWNLOADED {path.relative_to(args.output)} | {path.stat().st_size/1e6:.2f} MB | {rows} rows')
            current += timedelta(days=1)
    except KeyboardInterrupt:
        log('Interrupted. Rerun the same command to resume.')
        exit_code = 130
    except (AuthError, OSError) as exc:
        # AuthError text is controlled. Avoid printing paths/URLs from arbitrary network errors.
        note = str(exc) if isinstance(exc, AuthError) else f'Local I/O failure: {type(exc).__name__}; check free space and permissions'
        log('STOP: ' + note)
        counts['errors'] += 1
        if current_key:
            ledger.record(current_key, 'error', note=note)
        exit_code = 2
    finally:
        ledger.export()
        ledger.db.close()
        client.session.close()
        lock.close()
        log(f'SUMMARY {counts}')
        log('See rest_manifest.csv and rest_unavailable_or_errors.csv in the output folder.')
        log('Readable files do not prove complete market coverage. 404 is not zero liquidations.')
    return exit_code


if __name__ == '__main__':
    raise SystemExit(main())
