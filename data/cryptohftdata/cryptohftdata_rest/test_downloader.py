import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq
import requests
import zstandard as zstd
import download_cryptohftdata_rest as d


class Response:
    def __init__(self, status=200, body=b'', data=None, headers=None, fail=False):
        self.status_code, self.body, self.data = status, body, data
        self.headers = headers or {}
        self.fail = fail
    def __enter__(self): return self
    def __exit__(self, *args): pass
    def json(self): return self.data
    def iter_content(self, size):
        yield self.body[:10]
        if self.fail: raise requests.ConnectionError('private URL must not be logged')
        yield self.body[10:]


class Tests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.output = Path(self.tmp.name)
        stream = io.BytesIO()
        pq.write_table(pa.table({'price': [1., 2., 3.], 'side': ['BUY', 'SELL', 'BUY']}), stream)
        self.raw = stream.getvalue()
        self.key = 'binance_futures/2025-09-20/13/BTCUSDT_trades.parquet'
        self.client = d.Client('private-key', retries=3, interval=0)
        self.pause = patch.object(d, 'pause').start()
        self.logs = patch.object(d, 'log').start()
        self.post = patch.object(self.client.session, 'post', return_value=Response(data={'jwt_token':'test-token', 'expires_in':14400})).start()
    def tearDown(self):
        patch.stopall()
        self.client.session.close()
        self.tmp.cleanup()
    def test_auto_refresh(self):
        self.client.ensure_token()
        self.client.ensure_token()
        self.assertEqual(self.post.call_count, 1)
        self.client.refresh_at = 0
        self.client.ensure_token()
        self.assertEqual(self.post.call_count, 2)
    def test_auth_retry_same_file(self):
        with patch.object(self.client.session, 'get', side_effect=[Response(401), Response(body=self.raw)]) as get:
            path, rows = self.client.download(self.key, self.output)
        self.assertEqual(rows, 3)
        self.assertEqual(path.read_bytes(), self.raw)
        self.assertEqual(self.post.call_count, 2)
        self.assertEqual(get.call_args_list[0].kwargs['params'], get.call_args_list[1].kwargs['params'])
    def test_repeated_forbidden_stops(self):
        with patch.object(self.client.session, 'get', return_value=Response(403)):
            with self.assertRaises(d.AuthError): self.client.download(self.key, self.output)
    def test_rate_limit_and_server_retry(self):
        with patch.object(self.client.session, 'get', side_effect=[Response(429, headers={'Retry-After':'7'}), Response(503), Response(body=self.raw)]):
            self.client.download(self.key, self.output)
        self.pause.assert_any_call(7.)
    def test_unavailable(self):
        with patch.object(self.client.session, 'get', return_value=Response(404)):
            self.assertIsNone(self.client.download(self.key, self.output))
        self.assertFalse((self.output/self.key).exists())
    def test_stream_retry_and_atomic_commit(self):
        target = self.output/self.key
        target.parent.mkdir(parents=True)
        target.write_bytes(b'previous file')
        with patch.object(self.client.session, 'get', return_value=Response(body=self.raw, fail=True)):
            with self.assertRaises(RuntimeError): self.client.download(self.key, self.output)
        self.assertEqual(target.read_bytes(), b'previous file')
        self.assertFalse(list(self.output.rglob('*.part')))
        self.assertNotIn('private URL', str(self.logs.call_args_list))
    def test_wrong_length_not_committed(self):
        with patch.object(self.client.session, 'get', return_value=Response(body=self.raw, headers={'Content-Length':'1'})):
            with self.assertRaises(RuntimeError): self.client.download(self.key, self.output)
        self.assertFalse((self.output/self.key).exists())
    def test_zstd_download_and_local_resume(self):
        compressed = zstd.ZstdCompressor().compress(self.raw)
        with patch.object(self.client.session, 'get', return_value=Response(body=compressed)):
            path, rows = self.client.download(self.key, self.output)
        self.assertTrue(str(path).endswith('.parquet.zst'))
        self.assertEqual(rows, 3)
        ledger = d.Ledger(self.output)
        try:
            self.assertEqual(ledger.local(self.key), 'validated')
            with patch.object(d, 'parquet_rows', side_effect=AssertionError('must use cache')):
                self.assertEqual(ledger.local(self.key), 'cached')
            path.write_bytes(compressed[:20])
            self.assertIsNone(ledger.local(self.key))
            ledger.record(self.key, 'unavailable', note='HTTP 404')
            ledger.export()
            self.assertIn(self.key, (self.output/'rest_unavailable_or_errors.csv').read_text())
        finally: ledger.db.close()
    def test_corrupt_local_parquet(self):
        target = self.output/self.key
        target.parent.mkdir(parents=True)
        target.write_bytes(self.raw[:-10])
        ledger = d.Ledger(self.output)
        try: self.assertIsNone(ledger.local(self.key))
        finally: ledger.db.close()


if __name__ == '__main__': unittest.main()
