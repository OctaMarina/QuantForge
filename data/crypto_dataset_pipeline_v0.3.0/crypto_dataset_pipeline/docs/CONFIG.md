# Configurație

Valorile implicite sunt în `cryptods/config.py` (`DEFAULTS`); un fișier YAML le suprascrie parțial, iar
`--set cheie.cu.puncte=valoare` (repetabil) are prioritate. `python crypto_dataset.py show-config` afișează
configurația efectivă. Valorile implicite sunt **puncte de plecare rezonabile, nu valori optimizate**.

Identitatea unei serii de bare (`series_id`) depinde doar de: exchange, tip piață, simbol, tip bară, prag/interval,
`bar_origin` (bare de evenimente), sursa de tranzacții, politica de goluri, zecimalele de reprezentare și — pentru
information bars — exact parametrii `bars.information.*` care schimbă barele acelui tip (efectivi, după
`per_type`): runs: semn, `ewma_span_bars`, warm-up, limite / `fixed_expected_ticks`; imbalance AFML: în plus
`ewma_span_ticks` și podeaua; imbalance RMS: semn, `ewma_span_ticks`, warm-up, `rms_horizon_ticks`. Gărzile nu intră. Perioada de
export și warm-up-ul **nu** intră în identitate, deci nu mută granițele. Hash-ul complet al configurației
identifică build-ul (`data/datasets/<config_hash>/`).

| Cheie | Implicit | De ce |
|---|---|---|
| `instrument.exchange / market_type / base / quote` | binance / spot / BTC / USDT | cel mai lichid instrument cu arhivă publică de execuții individuale |
| `instrument.symbol` | derivat = base + quote | dacă e dat, **trebuie** să fie egal cu base + quote (sau + `_PERP` la perpetual); altfel validarea refuză configurația — schimbați activul din `base`/`quote` (ex. `configs/eth_spot.yaml`) |
| `instrument.price_decimals / qty_decimals` | din registrul intern (BTCUSDT: 2 / 8) | reprezentare întreagă exactă; o valoare cu mai multe zecimale ridică `PrecisionError` (nu se rotunjește) |
| `instrument.contract.type / contract_size` | linear / null | invers fără mărime de contract declarată = refuzat |
| `period.start / end` | 2024-06-01 / 2024-06-08 | o săptămână demonstrativă; interval semi-deschis UTC |
| `period.bar_origin` | 2024-05-25 | barele de evenimente se acumulează de aici; trebuie ≤ start |
| `warmup.intraday_days` | 31 | randament 30d / RV 30d necesită istoric anterior |
| `warmup.daily_days` | 200 | z-score 90 zile + trend 30 zile + marjă |
| `bars.types` | [time, volume, notional] | cele trei dataseturi; în plus pot fi cerute `tick_imbalance, volume_imbalance, notional_imbalance, tick_runs, volume_runs, notional_runs` (`INFORMATION_BARS.md`) |
| `bars.trade_source` | trades | execuții individuale (aggTrades doar ca alternativă declarată) |
| `bars.time.interval` | 5min | ~288 bare/zi |
| `bars.volume.threshold` | 100 (BTC) | ordinul a câteva sute de bare/zi la volumele BTCUSDT din 2024 (neoptimizat) |
| `bars.notional.threshold` | 10 000 000 (USDT) | idem în notional de cotare; moneda = quote (nu USD) |
| `bars.adaptive` | dezactivat | dacă e activat: prag = media volumului zilelor UTC complete anterioare / bare țintă, înghețat la începutul barei; prag inițial = pragul fix până există istoric |
| `bars.gap_policy` | invalidate_and_restart_segment | bara deschisă la gol este invalidată; un segment nou începe la prima tranzacție de după gol; `halt` oprește build-ul |
| `bars.max_ts_regression_us` | 0 | nicio regresie de timestamp în ordinea ID-urilor nu e tolerată |
| `bars.on_unconfirmed_missing_day` | stop | o zi de tranzacții care lipsește **fără** 404 confirmat (descărcare eșuată, cache incomplet, rețea) oprește construcția barelor înaintea ei, ca o rulare ulterioară să o poată completa; `gap` = acceptă explicit un gol permanent (comportamentul vechi) |
| `bars.archive_publication_lag_days` | 2 | un 404 pentru o zi mai recentă de atât nu e considerat „confirmat” (fișierele zilnice apar a doua zi); o zi confirmată lipsă e marcată în `raw/binance/<cale>.notfound.json` și devine gol |
| `bars.information.sign` | tick_rule | `b_t` din tick rule (AFML, cu `b_0` purtat din bara precedentă); `aggressor` = variantă separată cu agresorul observat |
| `bars.information.ewma_span_bars / ewma_span_ticks` | 20 / 10 000 | EWMA (adjust=True) pe barele anterioare (`E0[T]`, `P[b=1]`, `E0[v|b]`) / pe tranzacțiile barelor anterioare (`b_t v_t`) |
| `bars.information.warmup_ticks` | 10 000 | prima bară a fiecărui segment = bară de N tranzacții care inițializează așteptările; la runs bars stabilește practic scara barelor |
| `bars.information.expected_ticks_bounds` | null | stabilizator **non-AFML**: `[min, max]` pentru `E0[T]` |
| `bars.information.min_expected_imbalance_ratio` | null | stabilizator **non-AFML**: `|E[b v]| ≥ r · E[v]` (imbalance) |
| `bars.information.guard_max_bars_per_day` | 50 000 | garda de colaps: ziua e aruncată și seria oprită (cod ieșire 3); `null` dezactivează |
| `bars.information.guard_max_open_days` | 7 | avertisment dacă o bară rămâne deschisă mai mult (prag de neatins) |
| `bars.information.fixed_expected_ticks` | null | **non-AFML, recomandat pentru runs**: `E0[T] := H` constant (elimină deriva scării); exclusiv cu `expected_ticks_bounds` |
| `bars.information.imbalance_threshold` | afml | `afml` = `E0[T]·|E[b v]|` (literal); `rms` = **non-AFML, recomandat**: RMS empiric al dezechilibrului pe ferestre de H tranzacții |
| `bars.information.rms_horizon_ticks` | null | H pentru `rms` (multiplu de 4); obligatoriu cu `rms`; podeaua, limitele și `fixed_expected_ticks` nu se aplică la `rms` |
| `bars.information.per_type` | {} | parametri per tip, ex. `{tick_runs: {fixed_expected_ticks: 20000, warmup_ticks: 20000}, volume_imbalance: {rms_horizon_ticks: 12000}}`; un singur build poate produce toate tipurile, fiecare cu parametrii lui |
| `timing.replay_mode` | market_time | nivel 1 de dovadă (timestamp/secvență de exchange); `observed_availability` cere loguri de recepție |
| `timing.bar_finalization_delay / decision_latency` | 0ms / 0ms | ipoteze explicite; măriți-le pentru o politică de decizie realistă |
| `timing.cross_stream_tie` | exclude_equal | o observație din alt flux cu același timestamp ca decizia nu este considerată cunoscută |
| `timing.strictness` | strict | fără dovadă point-in-time ⇒ null; `research` adaugă coloane cu lag presupus într-un fișier separat |
| `timing.live_finalization_grace` | 2s | în modul observed, o bară de timp devine disponibilă la max(sfârșit + grace, recepția ultimei tranzacții) |
| `timing.research_assumed_lags` | etf 27h, F&G 3h | numai pentru research, niciodată certificat |
| `features.returns.calendar_horizons` | 5min, 1h, 4h, 1d, 7d, 30d | orizonturi cerute |
| `features.returns.bar_horizons` | 1, 5, 20 | randamente pe bare, separate de cele calendaristice |
| `features.returns.ref_tolerance` | 1min | vechimea maximă a prețului de referință la t−h |
| `features.realized_vol.grid / windows` | 1min / 1h, 1d, 7d, 30d | grilă explicită; 30d permite comparația descriptivă cu DVOL |
| `features.realized_vol.max_grid_staleness` | 5min | punct de grilă fără tranzacție recentă ⇒ invalid |
| `features.flow.min_trades_regression / bar_windows` | 20 / [50] | λ pe tranzacțiile barei / rolling pe bare |
| `features.structural.*` | W=200, τ=30, p=1, intercept, q=0.95, trim=0.15 | configurația testelor (declarată, neoptimizată); `eval_every` > 1 lasă null cu motivul `not_evaluated` |
| `features.premium.*` | Coinbase; BTC, ETH (+ activul); FX USDT-USD; acoperire zilnică ≥ 0.9; z 90 (min 60) | |
| `features.etf.*` | z 60 sesiuni (min 40); vechime max 4 zile | |
| `features.dvol.*` | BTC, ETH; rezoluție 60 s; vechime max 5 min | `max_age` trebuie să fie ≥ rezoluția (altfel orice lumânare ar fi „stale”; refuzat de validare). Pentru istoric lung: `resolution: "3600"`, `max_age: 2h` (retenția seriei de 1 min la Deribit este scurtă) |
| `features.skew.*` | 30 zile, delta 0.25, IV = mid(bid_iv, ask_iv), quote ≤ 60 s, snapshot ≤ 5 min | |
| `features.liquidations.*` | ferestre 5min, 1h; surpriză vs. 30 zile (min 500 ferestre) | |
| `features.book.bands_bps` | 10, 25, 50 | |
| `features.book.sizes_base / sizes_quote` | 0.1, 1, 10 BTC / 10k, 100k, 1M USDT | mărimile pentru cost; cantitatea pentru o mărime în quote = N / mid |
| `features.book.taker_fee_bps` | null | fără ipoteză de comision ⇒ coloana cu comision nu există, componentele rămân |
| `features.book.max_book_age / max_quote_age` | 2s / 2s | pentru un flux diff „nicio actualizare” poate însemna „fără schimbare”; măriți dacă sesiunea e liniștită |
| `features.book.wall.*` | bin 10 USDT (BTC), ≤ 100 bps, prag = 5 × mediana binurilor de pe aceeași parte, eșantionare 10 s | definiție explicită a „concentrărilor”; alternativ `history_quantile` |
| `storage.root` | ./data | toate nivelurile (raw / normalized / state / datasets / live) |
| `storage.max_download_gb` | 50 | dry-run-ul semnalează depășirea |
| `storage.rate_limit_per_s` | per furnizor | limitarea cererilor |
| `storage.request_timeout_s / retries / backoff_s` | 30 / 5 / 1.0 | retry exponențial pentru 429/5xx; refuzurile de politică (403/407) nu se reîncearcă |
| `storage.csv_export` | false | Parquet este formatul principal |
| `storage.omit_all_null_columns` | false | omiterea coloanelor complet nule doar explicit; lista e trecută în manifest |
| `storage.qa_kline_crosscheck` | true | bare interne de 1 min comparate cu klines native |
| `storage.keep_normalized_trades` | false | scrie și tranzacțiile normalizate pe zi (nivelul 2) |
| `api_keys.tardis_env` | TARDIS_API_KEY | cheia se citește doar din mediu, nu e stocată |
| `live.*` | toate fluxurile; diff depth 100 ms; rotație 1 h; poll 15 min | |

Opțiuni prezente dar **refuzate de validare** (neimplementate): `bars.usd_notional.enabled`,
`bars.native_kline_fallback`.

Avertismente (nefatale, afișate pe stderr și scrise în manifest la `warnings`): `bar_origin == start` cu bare de
evenimente (fără încălzire a ferestrelor pe bare), warm-up intraday mai scurt decât cea mai lungă fereastră
calendaristică, imbalance bars literale (fără stabilizatoare), moneda skew-ului diferită de activul de bază.

Coduri de ieșire ale CLI: `0` reușit (inclusiv „complet până la lag-ul de publicare”), `2` eroare de configurație,
`3` build incomplet (oprire înaintea unei zile neobținute, serie de information bars oprită de gardă, gol cu
`gap_policy=halt`, depășire int64).

Exemple:

```bash
python crypto_dataset.py dry-run --config configs/btc_demo.yaml --set period.end=2024-06-02T00:00:00Z
python crypto_dataset.py build   --config configs/btc_demo.yaml --set bars.types=[volume] --set bars.volume.threshold=50
python crypto_dataset.py build   --set instrument.market_type=perpetual --set instrument.price_decimals=2 --set instrument.qty_decimals=4
python crypto_dataset.py build   --set instrument.base=ETH --set features.book.wall.bin_size=0.5
python crypto_dataset.py build   --config configs/eth_spot.yaml --set bars.types=[volume,tick_runs,volume_runs]
```
