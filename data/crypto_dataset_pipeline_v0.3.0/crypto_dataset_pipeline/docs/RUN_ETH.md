# Cum generezi dataseturile ETHUSDT (pas cu pas)

Rezultatul final, pe perioada 2021-09-01 → 2026-09-01:

| Familie | Dataseturi | Parametri |
|---|---|---|
| volume bars | 3 | 19 600 / 9 800 / 4 900 ETH pe bară (~25 / 50 / 100 bare/zi) |
| notional bars | 3 | 48,3M / 24,15M / 12,075M USDT pe bară |
| runs bars (tick, volume, notional) | 3 × 3 | `E0[T]` fix, calibrat la 25 / 50 / 100 bare/zi |
| imbalance bars (tick, volume, notional) | 3 × 3 | prag RMS pe ferestre, calibrat la 25 / 50 / 100 bare/zi |

Metodologia barelor de informație și de ce diferă de forma literală AFML: [`INFORMATION_BARS.md`](INFORMATION_BARS.md).

## 1. Instalare (o singură dată)

```bash
cd crypto_dataset_pipeline
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt        # numba inclus: fără el rezultatele sunt identice, dar mult mai lente
.venv/bin/python -m pip install -r requirements-dev.txt && .venv/bin/python -m pytest -q   # opțional: testele
```

Python ≥ 3.11. Comenzile de mai jos se dau din directorul proiectului.

## 2. Dacă ai rulat versiuni mai vechi: curăță starea, păstrează descărcările

Build-urile anterioare versiunii 0.2.0 puteau îngheța în stare zile lipsă ca goluri permanente. Datele descărcate
(`data/raw`) rămân valabile și nu trebuie descărcate din nou:

```bash
rm -rf data/state data/normalized data/datasets data/_calibration calibration
```

## 3. Verifică ce se descarcă (opțional, recomandat)

```bash
.venv/bin/python crypto_dataset.py dry-run --config configs/eth_spot.yaml \
    --set period.start=2021-09-01T00:00:00Z --set period.end=2026-09-01T00:00:00Z \
    --set period.bar_origin=2021-08-01T00:00:00Z
```

Raportul arată zilele disponibile în arhiva Binance, volumul comprimat (GB) și accesul la fiecare sursă. Dacă
volumul depășește `storage.max_download_gb` (200 în `eth_spot.yaml`), mută datele pe un disc mare:
`EXTRA_SETS="storage.root=/cale/disc/mare" bash scripts/eth_datasets.sh`.

## 4. Rulează totul

```bash
nohup bash scripts/eth_datasets.sh > run_eth.log 2>&1 &
tail -f run_eth.log
```

Sau pas cu pas, în aceeași ordine:

```bash
bash scripts/eth_datasets.sh download      # 1. descărcare (reluabilă, CHECKSUM verificat)
bash scripts/eth_datasets.sh classic       # 2. cele 6 dataseturi volume / notional
bash scripts/eth_datasets.sh calibrate     # 3. calibrarea barelor de informație pe 2 luni pilot
bash scripts/eth_datasets.sh information   # 4. 3 build-uri, fiecare cu 6 tipuri de bare de informație
bash scripts/eth_datasets.sh report        # 5. tabelul cu toate dataseturile
```

Ce face fiecare pas:

1. **download** — trades zilnice ETHUSDT, plus sursele externe: klines, Coinbase, DVOL, funding, Fear & Greed,
   ETF. O descărcare întreruptă se reia de unde a rămas.
2. **classic** — volume și notional bars cu pragurile tale. Primul build parsează toate zilele și scrie numărul de
   tranzacții pe zi, pe care îl folosește calibrarea.
3. **calibrate** — pentru fiecare țintă (25/50/100) rulează luni pilot (2022-06 și 2024-03) și ajustează H până
   când lungimea medie a barelor e la ±5% de cea dorită. Rezultate: `calibration/calibration.json` (toate
   măsurătorile, inclusiv bare/zi așteptate pe fiecare an) și `calibration/target_25.args` etc. Seriile pilot stau
   în `data/_calibration` și pot fi șterse după calibrare.
4. **information** — pentru fiecare țintă, un singur build parcurge datele o dată și produce toate cele 6 tipuri,
   fiecare cu parametrii lui calibrați.
5. **report** — `data/reports/datasets_summary.csv` și un tabel pe ecran.

Fiecare pas poate fi rulat din nou fără să strice nimic: descărcările sunt în cache, iar build-urile continuă din
starea salvată și sar zilele deja procesate.

## 5. Dacă scriptul se oprește

Scriptul se oprește la prima eroare (`set -e`). Motivul apare în log:

| Mesaj | Cauză | Ce faci |
|---|---|---|
| `CONFIG ERROR: …` (cod 2) | parametru invalid | corectezi parametrul indicat |
| `INCOMPLETE: trades for <zi> not available (not_in_cache / download_failed)` (cod 3) | zi nedescărcată sau descărcare eșuată | `bash scripts/eth_datasets.sh download`, apoi reiei pasul oprit |
| `HALTED <tip>: …` (cod 3) | o serie a produs prea multe bare pe zi (garda de colaps) | nu ar trebui să apară cu variantele recomandate; trimite-mi `calibration.json` și mesajul |
| `BUILD STOPPED: OverflowError` (cod 3) | o bară ar depăși ~9,2·10⁸ USDT | nu apare la pragurile de aici; scade pragul / intervalul |
| `WARNING: series …: gap … now in the cache` | gol vechi în stare | ștergi `data/state/<series_id>` indicat și reiei pasul |

O zi care lipsește din arhiva Binance (404 confirmat) nu oprește build-ul: devine gol documentat, cu segment nou
(`manifest.json → missing_trade_days`, `segments`).

## 6. Verifică rezultatul

```bash
bash scripts/eth_datasets.sh report
```

Pentru fiecare dataset verifici:

* `status = complete`;
* `bars_per_day` apropiat de țintă. E media pe toată perioada; pe ani diferă după activitate (vezi
  `calibration.json → targets.<N>.expected_bars_per_day_by_year`), exact ca la volume bars;
* `qa_warning` gol și `halted` gol.

Pentru detalii pe un dataset: `data/datasets/<build>/manifest.json` → `datasets.<tip>.information_bar_qa`,
`warnings`, `missing_trade_days`, `segments`.

## 7. Cum folosești fișierele

Fiecare build are directorul `data/datasets/<config_hash>/`, cu `<tip>.parquet`, `<tip>_quality.parquet` (de ce
e null fiecare valoare), `<tip>_coverage.parquet`, `<tip>_registry.json` (fișa fiecărei coloane) și
`manifest.json`. Hash-ul nu spune nimic despre parametri; îi găsești în `datasets_summary.csv`:

```python
import pandas as pd
s = pd.read_csv("data/reports/datasets_summary.csv")
row = s[(s.bar_type == "volume_imbalance") & (s.bars_per_day.between(40, 60))].iloc[0]
df = pd.read_parquet(f"data/datasets/{row.build}/volume_imbalance.parquet")
```

Reguli de folosire corectă (detalii în `SCHEMA.md`):

* Un rând este o bară închisă. Pentru orice decizie folosești `decision_time`, nu `bar_end`.
* Valorile lipsă sunt null cu motiv (`<tip>_quality.parquet`), niciodată 0. Nu le completa înainte să vezi motivul.
* La barele de informație, rândurile cu `ib_warmup = true` sunt bare de inițializare. Cu originea pusă cu o lună
  înainte de start, ele cad în afara exportului.
* `bar_id` (= `series_id:bar_seq`) e cheia unică. Același `series_id` înseamnă aceleași granițe de bară, indiferent
  de perioada exportată.

## 8. Personalizare (variabile de mediu ale scriptului)

| Variabilă | Implicit | Rol |
|---|---|---|
| `START`, `END`, `ORIGIN` | 2021-09-01, 2026-09-01, 2021-08-01 | perioada exportată; originea barelor (o lună înainte) |
| `VOLUME_THRESHOLDS`, `NOTIONAL_THRESHOLDS` | pragurile tale | volume / notional bars |
| `TARGETS` | `25 50 100` | bare/zi țintă pentru barele de informație |
| `PILOT_MONTHS` | `2022-06,2024-03` | lunile de calibrare (în perioada descărcată) |
| `SIGN` | `aggressor` | `tick_rule` = aproximarea AFML a semnului |
| `CALIB_DIR` | `calibration` | unde se scriu rezultatele calibrării |
| `EXTRA_SETS` | gol | orice `--set` suplimentar, separate prin spațiu (ex. `storage.root=/mnt/date`) |
| `PYTHON`, `CONFIG` | `.venv/bin/python`, `configs/eth_spot.yaml` | |

Exemplu: doar ținta 50, recalibrată pe altă lună:
`TARGETS="50" PILOT_MONTHS="2023-05" bash scripts/eth_datasets.sh calibrate information report`.
