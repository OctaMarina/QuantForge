# Jurnalul verificării surselor (2026-10-20)

Accesul din mediul de build: PyPI și GitHub permise; **toate domeniile furnizorilor de date blocate** de proxy
(`CONNECT` → 403). Cererile WebFetch către endpointuri API au necesitat aprobare și nu au fost aprobate. Prin urmare,
verificările de mai jos provin din documentație (GitHub oficial sau pagini publice de documentație) și din
eșantioane reale disponibile pe GitHub, nu din răspunsuri live ale API-urilor.

| Sursă | Ce s-a verificat | Unde | Ce NU s-a putut verifica |
|---|---|---|---|
| Binance public data | căi `data/{spot,futures/um,futures/cm}/{daily,monthly}/<tip>/<SIMBOL>/…zip`, fișier `.CHECKSUM` sha256, coloanele trades/aggTrades/klines, „spot din 2025-01-01 în microsecunde”, „fișierele arhivate pot fi actualizate ulterior” (2 seturi de actualizări listate) | `github.com/binance/binance-public-data` (README, `python/utility.py`, `enums.py`) | existența efectivă a fiecărei zile, dimensiunile, prima zi disponibilă (se verifică la `dry-run` prin listarea S3) |
| Binance spot API/WS | stream `@trade` (t, p, q, T, m), `@depth` (U, u, b, a), procedura oficială de carnet local (buffer, snapshot ≤ 5000 niveluri, regula U ≤ lastUpdateId+1 ≤ u, resync la gol), `@bookTicker` fără timestamp, `timeUnit=MICROSECOND`, conexiune max 24 h, ping 20 s; `historicalTrades` fără cheie API; block trades din 2026-05 cu endpoint/stream separat | `github.com/binance/binance-spot-api-docs` (web-socket-streams.md, rest-api.md, CHANGELOG.md, actualizat 2026-09-18) | includerea block trades în arhiva `trades` |
| Binance USD-M | `forceOrder`: „only the latest one liquidation order within 1000ms will be pushed as the snapshot”; câmpuri S, q, p, ap, X, l, z, T; diff depth cu `pu` și regula de continuitate; endpointuri `fundingRate`, `fundingInfo`, `premiumIndexKlines`, `markPriceKlines`, `indexPriceKlines` | `github.com/binance/binance-futures-connector-python` (docstring-uri oficiale), documentația testnet derivate | coloanele arhivelor `fundingRate`/`bookTicker` (parserul citește antetul și refuză layout-uri necunoscute) |
| Bybit v5 | `allLiquidation.{symbol}`: push 500 ms, „S: Position side… Buy… long position has been liquidated”, v = executed size, p = bankruptcy price; orderbook: snapshot/delta, mărimi absolute, 0 = ștergere, u = 1 ⇒ restart, RPI excluse; URL `wss://stream.bybit.com/v5/public/linear`, ping `{"op":"ping"}` | `github.com/bybit-exchange/docs` (commit din 2026-10-14) | — |
| Coinbase Exchange | `/products/{id}/candles`: granularități {60,…,86400}, max 300 lumânări/cerere, `time` = începutul bucket-ului, „No data is published for intervals where there are no ticks” | docs.cdp.coinbase.com | unitatea volumului; existența istoricului USDT-USD pe perioada cerută |
| Deribit | `get_volatility_index_data(currency, start, end, resolution∈{1,60,3600,43200,1D})`, `continuation`; `ticker`: bid_iv, ask_iv, mark_iv, greeks.delta („Black Scholes Delta”) | docs.deribit.com | unitatea DVOL (API-ul nu o declară; articolul de metodologie Deribit: „30 day annualised implied volatility”, exemplul DVOL=90 ⇒ ~4.5%/zi ⇒ puncte procentuale); adâncimea istorică; dacă timestamp-ul lumânării e început sau sfârșit (tratat conservator ca început) |
| Alternative.me | `GET /fng/?limit=0`, câmpuri value/value_classification/timestamp/time_until_update; metodologie centrată pe Bitcoin | alternative.me/crypto/fear-and-greed-index | ora publicării, politica de revizuire (nu sunt documentate) |
| Farside | tabel „Bitcoin ETF Flow (US$m)”, coloane pe fond + Total, paranteze pentru negative, rânduri Fee/Total/Average/Maximum/Minimum, doar sesiuni de tranzacționare | farside.co.uk/btc/ (citit prin WebFetch) | momentul primei publicări a fiecărei valori și revizuirile (de aceea modul strict cere vintage-uri) |
| Tardis.dev | URL `datasets.tardis.dev/v1/{exchange}/{type}/{YYYY}/{MM}/{DD}/{symbol}.csv.gz`, fără cheie doar prima zi a lunii, coloanele tuturor tipurilor, `local_timestamp` = momentul sosirii, `side` lichidări: „buy - short position was liquidated”, `amount` „as provided by exchange” | docs.tardis.dev | unitatea IV și convenția delta din options_chain (nedocumentate) |
| AFML 2018 | cuprinsul capitolului 17 (17.3 CUSUM Tests, 17.4 Explosiveness Tests) | O'Reilly (previzualizare) | **formulele** (necitibile în previzualizare); capitolele 2 și 19 |
| MizarLabs | sursele `sadf.py` și `sdfc.py` (comparație de implementare) | mizarlabs.readthedocs.io | — (nu e autoritate pentru formulă) |
| statsmodels 0.15.0 | `recursive_olsresiduals`: rresid_scaled, var(ddof=1) pe rresid_scaled[skip:], rcusum | sursa instalată | — |

## Eșantioane reale folosite (GitHub)

`scripts/fetch_real_fixtures.sh` descarcă, de la commit-ul fixat `16de62c98b59c322129f9a85c98b8c9c6bb52b04` al
repository-ului `nautechsystems/nautilus_trader` (LGPL-3.0), directorul `test_data`:

* `binance/ethusdt-trades.csv` — 69 806 rânduri Binance ETHUSDT spot, 2020-08-14 10:00:00.223–14:59:58.693 UTC;
  70 de rânduri sunt duplicate exacte (suprapunere de paginare), eliminate după `trade_id`; după deduplicare ID-urile
  148568980…148638715 sunt contigue;
* `binance/btcusdt-trades.parquet` și `btcusdt-quotes.parquet` — 2 001 tranzacții și 451 de cotații BTCUSDT,
  2021-01-08 00:00:00–00:00:46 UTC;
* `binance/btcusdt-depth-snap.csv` / `-update.csv` — snapshot (100 niveluri bid) și un diff Binance USD-M BTCUSDT,
  2022-11-01; diff-ul nu acoperă `lastUpdateId` al snapshot-ului, iar reconstrucția îl refuză (gol), cum cere procedura;
* `tardis/deribit_incremental_book_L2_BTC-PERPETUAL.csv` — L2 incremental Deribit (contract invers, cantități în USD).

Acestea sunt copii terțe; nu au putut fi comparate cu fișierele `.CHECKSUM` Binance din acest mediu.
