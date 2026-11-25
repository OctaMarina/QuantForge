# chdds — dataseturi ML reproductibile din microstructura BTCUSDT perpetual (Binance USDⓈ-M, CryptoHFTData)

Biblioteca construiește, din fișierele orare CryptoHFTData (trades, orderbook L2, ticker, liquidations,
open_interest, mark_price), **time / volume / dollar (notional USDT) / CUSUM bars** și **features cauzale** din
toate cele șase surse, cu măști de validitate, proveniență și manifest. Țintele/etichetele **nu** fac parte din
bibliotecă (separare intenționată).

Documente:
- `docs/AUDIT_REPORT.md` — auditul istoricului complet (ce este exhaustiv, ce este eșantion), decizii, blocaje.
- `docs/AUDIT_TABLES.md` — tabelele lunare generate automat (`chdds audit-report`), CSV-uri în `derived/audit/`.
- `docs/VALIDITY_MAP.md` — harta intervalelor valide pe familie.
- `docs/FEATURE_REGISTRY.md` + `docs/feature_registry.csv` — registrul definițiilor (generat din `chdds/registry.py`).
- `docs/ARCHITECTURE.md` — straturi, ceasuri, convenții de bare, reconstrucția L2.
- `docs/RUNBOOK.md` — comenzi, reluare, extindere, alte configurații; `docs/BENCHMARK.md`; `docs/LIMITATIONS.md`.
- `docs/D1_REPORT.md` — features de book fără ancoră: rezultate și recomandare; `docs/D1_TABLES.md` — tabelele generate.
- `docs/TEST_RESULTS.md` — rezultatele testelor.
- `docs/vendor_docs_2026-11-03/` — copia documentației furnizorului consultată (versionare).

## Instalare (mediu virtual)

```bash
cd ~/Desktop/QuantForge/data/cryptohftdata
python3 -m venv .venv
.venv/bin/pip install -r requirements-lock.txt     # versiunile exacte folosite
.venv/bin/pip install -e .
.venv/bin/python -m pytest -q                       # suita de teste
```

Datele brute rămân în `cryptohftdata_raw/` și nu sunt modificate; tot ce produce biblioteca este în `derived/`
(variabile de mediu: `CHDDS_RAW_ROOT`, `CHDDS_DERIVED`).

## Comenzi (toate reluabile)

```bash
C=configs/btcusdt_initial.yaml
.venv/bin/chdds audit --workers 9          # audit exhaustiv per fișier + stratul "staged" (întregi zecimali exacți)
.venv/bin/chdds consolidate                # tabele de audit
.venv/bin/chdds canonical                  # trades canonice + mark/ticker/OI/lichidări
.venv/bin/chdds bars --config $C           # bare (incremental: continuă după ultima zi construită)
.venv/bin/chdds features --config $C       # features din trades + fluxuri lente (paralel pe zile)
.venv/bin/chdds book --config $C           # un singur replay L2 cauzal pentru toate seturile de bare
.venv/bin/chdds assemble --config $C       # join + combinații + selecție coloane + manifest
.venv/bin/chdds audit-report               # docs/AUDIT_TABLES.md
.venv/bin/chdds registry                   # docs/FEATURE_REGISTRY.md
```

Orice parametru: `--set dataset.end=2026-10-31 --set features.book.max_age_ms=500`.
Rezultat: `derived/datasets/<nume>/<spec_id>/YYYY-MM.parquet` + `manifest.json` (config, hash cod, versiuni).

## Ce s-a produs (rularea din 2026-11-03/04)

| | |
|---|---|
| Audit exhaustiv | 70.354 fișiere, 60.511.118.623 rânduri, 0 erori de citire (`docs/AUDIT_REPORT.md`) |
| Tranzacții economice canonice | 1.771.277.674 (5,41 mil. copii, 3,48 mil. NA, 30.366 INSURANCE_FUND zero eliminate; 2.846 goluri de ID) |
| Seturi de bare (`configs/btcusdt_initial.yaml`) | time 1 m / 5 m; volume 50 BTC / adaptiv 60 s; dollar 5 M USDT / adaptiv 60 s; CUSUM 5 / 10 bps / adaptiv c=1 — 9.413.305 bare, conservare exactă |
| Dataset | `derived/datasets/btcusdt_initial/<spec_id>/YYYY-MM.parquet` (664–670 coloane) + `manifest.json` + `verification.json` |
| Demo verificat | `derived/datasets/demo_3days/` (2025-06-28 → 30, inclusiv bare de 5 s și CUSUM „both”, book valid și verificat) |
| Book L2 | `btcusdt_initial`: 2.259 verificări, valid strict 26,2%. `btcusdt_v2`: 3.314 verificări, 0 nepotriviri, vârf de book 97,3% pe bare de 1 minut (`docs/DATASET_V2.md`) |
| Dataset curent | `derived/datasets/btcusdt_v2/`, `book_mode: rebuilt`; `book_state` și proveniența sunt metadate de calitate, excluse implicit din features pentru model |
| Teste | 77 (`docs/TEST_RESULTS.md`) |

Limitări: `docs/LIMITATIONS.md`. Regenerare: `docs/REGEN_READINESS.md`. Stadiul implementat/testat/executat: `docs/STATUS.md`.

Citire rapidă: `examples/load_dataset.py`; toate variantele de bare: `configs/examples_all_bar_types.yaml`.
