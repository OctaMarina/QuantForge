# Schema datelor și a ieșirilor

## Niveluri de stocare (`storage.root`)

| Nivel | Cale | Conținut |
|---|---|---|
| 1. brut | `raw/<sursă>/<cale>` + `.meta.json` | fișierele exact cum au fost primite (zip-uri Binance cu CHECKSUM verificat, pagini JSON/HTML), URL, sha256, antete HTTP, `ingested_at`, versiune; o revizie nu suprascrie: `.v2`, `.v3` + `_revisions.jsonl` |
| 1. live | `live/<flux>/<zi>/<HHMM>.jsonl.gz` | fiecare mesaj WebSocket / poll cu `received_at_us`; `live/_conn` = deschideri/închideri de conexiune (acoperire) |
| 2. normalizat | `normalized/px1s/…`, `normalized/trades_quality/…`, `normalized/coinbase/…`, `normalized/deribit/…`, opțional `normalized/trades/…` | tabele reconstruite (prețul previous-tick pe secundă, calitatea fiecărui fișier zilnic, serii externe) |
| 2. stare | `state/<series_id>/state.json`, `state/<series_id>/bars/part-*.parquet` | starea builderului (inclusiv bara deschisă, sumele float în hex exact), barele închise |
| 3. final | `datasets/<config_hash>/` | dataseturi per tip de bară + manifest + registre + calitate + acoperire |

## Fișiere în `datasets/<config_hash>/`

| Fișier | Descriere |
|---|---|
| `<bar>.parquet` | datasetul final (un rând = o bară **închisă**, în perioada de export) |
| `<bar>_partial.parquet` | bara curentă neînchisă (niciodată etichetată finală) |
| `<bar>_invalidated_bars.parquet` | bare de evenimente tăiate de un gol (nu sunt bare) |
| `<bar>_quality.parquet` | motive de null comprimate în rulaje: `feature, reason, first_row, last_row, n_rows` (`*_row` = `bar_seq`) |
| `<bar>_coverage.parquet` | acoperire pe feature × lună: `rows, valid, valid_share, internal_null_runs, null_reasons` |
| `<bar>_registry.json/.md` | fișa fiecărei coloane (formulă, unitate, clock, ferestre, intrări, furnizor, disponibilitate, staleness, natură, motive null, eligibilitate strictă, referință, statut) |
| `source_coverage.parquet` | acoperire pe sursă × lună, inclusiv minute lipsă interne pentru seriile de 1 min |
| `observations/*.parquet` | observațiile externe cu contractul temporal complet (published_at / received_at / ingested_at / available_at / versiune / dovadă) |
| `research/<bar>_assumed_lag.parquet` | numai în modul research: coloane cu lag presupus, `point_in_time_certified = false` |
| `manifest.json` | configurație, hash-uri (config, cod, registru), dependențe, endpointuri, data verificării documentației, intrări (sha256), segmente/goluri (rupturi), statutul surselor și familiilor, verificarea vs. klines, invarianța față de build-ul anterior |
| `<bar>.prev.parquet` | copia build-ului anterior, folosită la `update` pentru verificarea invarianței prefixului |

## Coloane de bază ale datasetului

| Coloană | Tip | Semnificație |
|---|---|---|
| `bar_id` | string | `<series_id>:<bar_seq>` — unic chiar dacă mai multe bare se închid la același timestamp |
| `series_id`, `bar_type`, `bar_seq`, `segment_id` | | identitatea seriei; `segment_id` crește după fiecare gol nerecuperabil |
| `bar_start_us`, `bar_end_us` (+ `bar_start`, `bar_end` timestamp UTC) | int64 µs | time bars: [t, t+Δ); bare de evenimente: închiderea precedentă (sau originea segmentului) → timestamp-ul tranzacției care închide bara |
| `bar_available_us`, `decision_time_us` (+ variante timestamp) | int64 µs | `available = end + bar_finalization_delay` (market-time) sau din recepție (observed); `decision = available + decision_latency` |
| `first_trade_ts_us`, `last_trade_ts_us`, `first_trade_id`, `last_trade_id` | | intervalul de execuții al barei |
| `open, high, low, close` | float | prețuri executate; null pentru bare fără tranzacții și pentru bare incomplete |
| `volume` | float | unități de bază |
| `notional_<quote>` | float | Σ p·q în moneda de cotare (ex. `notional_usdt`) — nu USD |
| `n_trades`, `n_messages` | | execuții individuale / mesaje (diferite doar pentru aggTrades) |
| `threshold_frozen` | float | pragul înghețat la începutul barei (bare de evenimente) |
| `complete`, `bar_status` | | `closed`, `empty_interval` (feed complet, zero tranzacții), `incomplete_interval` |
| `bar_provenance` | | `trades` sau `aggTrades` |

Toate celelalte coloane sunt feature-uri descrise în `docs/FEATURES.md` și în registrul JSON al build-ului.
Valorile lipsă sunt `NaN`/null (niciodată 0); motivul fiecărui null este în `<bar>_quality.parquet`.

Vocabularul motivelor de null: `not_available_in_period, not_applicable, missing_source_data, unknown_publication_time,
unknown_historical_vintage, insufficient_history, stale_input, feed_gap, insufficient_book_depth,
unsupported_granularity, invalid_regression, unverified_formula, zero_denominator, nonpositive_log_input,
incomplete_coverage, source_unreachable, unsupported_instrument, not_finalized, invalid_quote, no_observation,
not_evaluated`.
