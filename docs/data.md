# `quant_forge.data` — market data

Download Binance spot data, keep it in local parquet caches, and turn it into the bar frame used by `quant_forge.strategies`
and `quant_forge.features`: **time bars** from klines and **volume / dollar bars** from every aggTrade, through one interface.

```python
from quant_forge.data import load_bars, BarConfig, SEEN_ASSETS, NEW_ASSETS, ALL_ASSETS, TICK_SIZES

bars = load_bars(ALL_ASSETS, BarConfig("time", 6), end="2026-11-19")      # 4h klines (30 days after listing dropped)
bars = load_bars(ALL_ASSETS, BarConfig("volume", 48), end="2026-11-19")   # volume bars, 48 per day on average
bars = load_bars(ALL_ASSETS, BarConfig("dollar", 1), end="2026-11-19")    # dollar bars, one per day on average
bars["BTC"].tail()
```

## Modules

### `bars` — bar frames and the unified loader

`BarConfig(bar_type, bars_per_day, window_days=30, max_gap_hours=4.0)` describes how bars are sampled:

| Field | Meaning |
|---|---|
| `bar_type` | `"time"`, `"volume"` or `"dollar"` |
| `bars_per_day` | time bars: 24 h / bar length, a Binance interval (48 → 30m, 24 → 1h, 6 → 4h, 1 → 1d, also 1440, 288, 96, 12, 4, 3, 2); event bars: the target average |
| `window_days`, `max_gap_hours` | event bars only (threshold window and outage gap, see below) |

Properties: `interval` (time bars), `event_spec` (event bars), `key` (`time_4h`, `volume_6`, `dollar_48`), `label`,
`nominal_hours`. `load_bars(assets, config, *, end, warmup_days=30, update=True)` returns `{asset: frame}` in the order of
`assets`, downloading or building what is missing when `update` is true.

`klines_to_bars(klines, *, bar="4h", end=None, warmup_days=30, seq_origin="2017-01-01")` returns, indexed by the bar end
(`t`, tz-aware UTC):

| Column | Meaning |
|---|---|
| `bar_start`, `bar_end` | UTC timestamps |
| `open`, `high`, `low`, `close`, `volume`, `quote_volume`, `n_trades` (float), `taker_buy_base` | kline fields |
| `bar_start_us`, `bar_end_us` | integer microseconds (strategy-library convention; computed with an explicit unit so pandas 3's millisecond resolution does not change them) |
| `bar_seq` | position on the regular grid; a jump means missing bars, which splits indicator segments in the strategy library |
| `complete` | 1 if the bar had at least one trade (bars without trades are invalid for the strategies) |

`load_binance_bars(assets, *, end, interval="4h", quote="USDT", ...)` is the time-bar loader behind `load_bars`.

### `binance` — REST client and kline cache

| Function | Purpose |
|---|---|
| `fetch_klines(symbol, interval="4h", start="2017-08-01", end=None)` | pages through `GET /api/v3/klines` (no API key) until `end`; price/volume columns as floats, `open_time`/`close_time` in ms |
| `fetch_klines_hybrid(symbol, interval, start, end)` | the public archive for the bulk of the history, REST for the days after its last file (the right source for 1-minute klines) |
| `load_klines(symbol, interval, *, end, start, cache_dir=None, update=True, source=None)` | reads `<cache_dir>/<SYMBOL>.parquet`; if the last cached kline closes before `end` it downloads **only from the last cached kline onwards** and replaces the overlap; `source="archive"` uses `fetch_klines_hybrid` |
| `interval_timedelta("4h")` | bar length of a Binance interval code |

The default cache directory is `~/.cache/quantforge/binance_klines_<interval>`.

### `archive` — the Binance public data archive (`data.binance.vision`)

One zip per symbol and month (and per day for the current month), each with a SHA-256 `.CHECKSUM`.

| Function | Purpose |
|---|---|
| `plan_archive_files(start, end, today=None)` | the monthly and daily files covering whole UTC days `[start, end)`: a month entirely inside the range and already finished is one monthly file, the other days are daily files (only days before `today`) |
| `stream_archive_batches(kind, symbol, period, columns=..., types=...)` | one file as Arrow record batches **while it downloads**: the zip entry (deflate, Zip64 for the 16 GB BTC months) is inflated on the fly, a header line is detected, nothing is written to disk; raises `ArchiveFileMissing` (404) or `ChecksumError` after the last batch |
| `fetch_klines_archive(symbol, interval, start, end)`, `fetch_aggtrades_archive(symbol, days)`, `fetch_archive_csv(...)` | whole files into DataFrames (small requests) |
| `normalize_epoch_ms(values)` | the archive switched to microsecond timestamps on 2025-01-01; everything is normalised to milliseconds |

### `event_bars` — volume and dollar bars from aggTrades

A bar closes when the base volume (`"volume"`) or quote volume (`"dollar"`) traded since it opened reaches a threshold.
**The threshold is fixed when the bar opens and uses only activity observable at that moment:**

    threshold(bar opening on UTC day D) = mean daily traded measure over the complete days D-W .. D-1 / bars_per_day

so a day as active as the trailing W days gives `bars_per_day` bars on average, and the realised frequency rises and falls
with market activity. No bar opens during the first W days of an asset (the warm-up). Conventions:

* trades with the same timestamp stay in one bar (a bar that reached its threshold closes when the next trade has a strictly
  later timestamp), so bar starts strictly increase and each bar starts after the previous one ended;
* `bar_start` = first trade, `bar_end` = last trade = the decision time; `first_trade_ts_us = bar_start_us`, so the
  strategies' `next_open` orders fill at the first trade of the next bar;
* a pause longer than `max_gap_hours` between two trades (outage, halt) closes the open bar (`closed_by = "gap"`) and starts a
  new `segment_id` — indicators restart and positions are censored, as at a missing kline;
* the bar still forming at the data end is not emitted; `n_trades` counts underlying trades (as klines do);
* records with a non-positive price or quantity are dropped and counted (`n_dropped`): the archive holds zero-price
  aggTrades of BTC, ETH, BNB and LTC on 2017-12-04 and in January 2018 that the klines do not contain;
* repeated records are dropped by aggTrade id (`n_duplicates`): a record is a duplicate when its id is not above every
  id seen and its timestamp is not newer than every timestamp seen (a smaller id with a newer timestamp restarts the
  sequence). The ETHUSDT 2018-01 file holds parts of the month four times (14 million duplicates; treating them as
  out-of-order trades would pile them into one instant and inflate the next 30 days' thresholds), and BTC and XRP
  repeat some trades of 2026-02-11. After both rules the daily totals equal the 1-day klines (worst 4e-6);
* the result does not depend on how the trade stream is cut into batches (bitwise; tested), and bars built from a prefix of
  the trades are a prefix of the bars (no look-ahead; tested).

```python
from quant_forge.data import EventBarSpec, DEFAULT_EVENT_SPECS, build_event_bars_many, load_event_bars, event_cache_status

build_event_bars_many(["BTCUSDT", "ETHUSDT"], DEFAULT_EVENT_SPECS, end="2026-11-19", workers=6)   # hours for BTC
bars = load_event_bars(["BTC", "ETH"], EventBarSpec("dollar", 6), end="2026-11-19", update=False)
event_cache_status("BTCUSDT")      # specs, next day to process, trades, archive bytes, files missing from the archive
```

| Object | Purpose |
|---|---|
| `EventBarSpec(kind, bars_per_day, window_days=30, max_gap_hours=4.0)` | one series; `name` = `dollar_6pd_w30_g4h`; `DEFAULT_EVENT_SPECS` = volume and dollar at 48, 24, 6, 1 per day |
| `EventBarBuilder(specs)` | streaming construction of several series: `feed(t_ms, price, qty, n_trades, is_buyer_maker)`, `take_bars()`, `snapshot()` / `restore()` |
| `event_bars_from_trades(trades, specs)` | raw bars of an in-memory trade table |
| `daily_thresholds(daily_measure, bars_per_day, window_days, first_day)` | the threshold rule, exposed for documentation and tests |
| `build_event_bars(symbol, specs, *, end)` | one chronological pass over the archive files of a symbol, all specs at once; after every file the bars (`<cache>/<SYMBOL>/<spec.name>.parquet`) and the builder state (`state.pkl`) are committed atomically, so an interrupted build resumes where it stopped and a later call processes only the new days (an incremental build equals a one-shot build; tested). A missing monthly file after the listing falls back to its daily files; files missing from the archive are recorded |
| `build_event_bars_many(symbols, specs, *, end, workers)` | the same for several symbols in parallel processes (largest first), progress in `<cache>/build.log` |
| `load_event_bars(assets, spec, *, end, update=True)` | `{asset: frame}`; builds or extends the cache first when needed |
| `event_bars_to_frame(raw, end=None)` | raw cached bars → bar frame (the time-bar columns plus `first_trade_ts_us`, `segment_id`, `threshold`, `taker_buy_quote`, `n_agg`, `closed_by`, `duration_h`, `bar_type`; `bar_seq` consecutive) |

Cache: `~/.cache/quantforge/binance_event_bars/`. The builder's daily totals equal the daily volumes of the 1-day klines to
~1e-14 (the check in the bar-type notebook), which shows the trade stream is complete. Cost: building the eight default series
for the 40 research assets reads ≈ 230 GB of zipped archive (≈ 16 billion aggTrades, about half of it BTC and ETH); the kernel
handles ~200 M trade-series per second, so the build is bound by the download (≈ 10 MB/s per connection).

### `diagnostics` — achieved frequency and bar durations

| Function | Purpose |
|---|---|
| `bar_durations(bars)` | hours since the previous bar end inside the same segment (NaN at a segment start) |
| `sampling_summary(bars, target_bars_per_day)` | one row per asset: bars, achieved bars per day and its ratio to the target, segments, duration quantiles, share of bars longer than 4x the nominal length |
| `sampling_by_period(bars, freq="YS")` | one row per (asset, period): bars, active days, bars per day, duration p10 / p50 / p90 |

### `universe` — research universes

* `SEEN_ASSETS` – 20 assets that the meta-models of the research notebooks train on;
* `NEW_ASSETS` – 20 assets used only for out-of-sample transfer (never trained on);
* `ALL_ASSETS = SEEN_ASSETS + NEW_ASSETS`, `TICK_SIZES` (required by the strategy library, relevant only with slippage).

## Notes

* All 40 assets still exist on Binance (survivorship bias of the universe).
* The REST endpoint is rate limited; `fetch_klines` pauses 0.15 s between pages.
* To use another venue, produce a DataFrame with the kline columns of `binance.KLINE_COLUMNS` and call `klines_to_bars`, or
  feed its trades to `EventBarBuilder` / `event_bars_from_trades` and convert with `event_bars_to_frame`.
* Strategy rules that read bar volume behave differently on volume bars, where every bar carries about the threshold volume
  (e.g. SuperTrend's volume filter); rules that rebuild calendar candles from the input bars (TIS's daily confirmation) need
  bars that respect day boundaries. The bar-type notebook quantifies both.
