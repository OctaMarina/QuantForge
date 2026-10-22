# cryptods — dataseturi crypto cauzale cu time, volume și quote-notional bars

Program Python reproductibil care descarcă și păstrează datele unei criptomonede, construiește exact **time bars,
volume bars și dollar/notional bars** din execuții individuale și calculează feature-urile cerute (familiile AFML
cap. 2/17/19, piețe încrucișate, date externe, L2, costuri), cu un contract temporal explicit împotriva lookahead-ului.

Ce **nu** face: nu antrenează modele, nu optimizează strategii, nu afirmă că vreun feature are alpha, nu plasează ordine.
Nu generează randamente viitoare sau etichete triple-barrier.

> **Starea în mediul în care a fost construit (2026-10-20):** toate domeniile furnizorilor de date au fost blocate de
> proxy, deci pipeline-ul nu a putut descărca istoric real. Au fost rulate: 80 de teste (inclusiv suita anti-lookahead pe
> tranzacții reale Binance ETHUSDT), un dataset real mic (`sample_output/`), plus `dry-run` / `download` / `build` /
> `live`, care au raportat corect sursele blocate. Detalii: [`docs/STATUS.md`](docs/STATUS.md).

## Instalare

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements-dev.txt         # versiunile exacte cu care au rulat testele (Python 3.13)
bash scripts/fetch_real_fixtures.sh         # eșantioane reale mici pentru teste / sample-real (GitHub)
python -m pytest -q                         # 80 teste
```

`numba` este opțional: fără el rezultatele sunt aceleași (verificat bit cu bit pe un eșantion), doar mai lente.

## Comenzi

```bash
python crypto_dataset.py dry-run     --config configs/btc_demo.yaml   # surse, acces, zile disponibile, GB estimați, blocaje
python crypto_dataset.py download    --config configs/btc_demo.yaml   # descarcă în cache (reluabil, verifică CHECKSUM)
python crypto_dataset.py build       --config configs/btc_demo.yaml   # construiește din cache, fără rețea
python crypto_dataset.py build       --config configs/btc_demo.yaml --fetch   # completează din rețea ce lipsește
python crypto_dataset.py update      --config configs/btc_demo.yaml   # zile noi + build incremental (+ verificare invarianță)
python crypto_dataset.py live        --config configs/btc_demo.yaml   # colectare live: trades, L2, bookTicker, lichidări, DVOL, opțiuni, poll-uri
python crypto_dataset.py replay-live --config configs/btc_demo.yaml   # dataseturi din logurile live (aceleași transformări)
python crypto_dataset.py sample-real --config configs/sample_real_ethusdt.yaml   # dataset real mic din fixture
python crypto_dataset.py registry    --config configs/btc_demo.yaml --fixtures out/   # registrul coloanelor, fără date
python crypto_dataset.py show-config --config configs/btc_demo.yaml
```

Orice parametru se schimbă cu `--set`, de exemplu:

```bash
python crypto_dataset.py build --set period.start=2024-03-01T00:00:00Z --set period.end=2024-04-01T00:00:00Z \
       --set period.bar_origin=2024-02-20T00:00:00Z --set bars.volume.threshold=200 --set bars.types=[volume,notional]
python crypto_dataset.py build --set instrument.market_type=perpetual          # BTCUSDT USD-M (liniar)
python crypto_dataset.py build --set timing.strictness=research                # + coloane cu lag presupus, fișier separat
```

Toate cheile și motivarea valorilor implicite: [`docs/CONFIG.md`](docs/CONFIG.md).

## Flux tipic

1. `dry-run` înainte de o descărcare mare: listează zilele disponibile în arhiva Binance (listare S3), dimensiunea
   comprimată, zilele lipsă, accesul la fiecare furnizor și ce poate livra fiecare familie.
2. `download` → `build`. Rezultatul: `data/datasets/<config_hash>/{time,volume,notional}.parquet` + manifest,
   registre, calitate, acoperire ([`docs/SCHEMA.md`](docs/SCHEMA.md)).
3. `live` în paralel, permanent: este singura cale gratuită pentru L2, lichidări, skew 25Δ și pentru dovada
   point-in-time a publicărilor (ETF, Fear & Greed). `replay-live` reconstruiește din loguri.
4. `update` periodic: builderele reiau din starea salvată (inclusiv bara deschisă); rândurile deja publicate sunt
   comparate cu build-ul anterior (`prefix_invariance_vs_previous_build` în manifest).

## Contractul temporal (pe scurt)

* Un rând = o bară **închisă**. Se stochează separat `bar_end`, `bar_available` și `decision_time`.
* Două niveluri de dovadă, niciodată amestecate: **market-time replay** (timestamp/secvență de exchange; nu se
  pretinde latență live măsurată) și **observed-availability replay** (din logurile de recepție ale colectorului).
* Îmbinări numai backward as-of pe `available_at`; o observație din alt flux cu același timestamp ca decizia nu e
  considerată cunoscută. Vechimea e livrată și limitată; o sursă lentă nu e înlocuită de o observație viitoare.
* Lumânările etichetate prin deschidere (Binance, Coinbase, DVOL) devin disponibile abia la sfârșitul intervalului.
* Publicări externe: în modul `strict`, fără dovadă a momentului publicării (poll live cu `received_at`, captură
  Wayback) valoarea eligibilă este null (`unknown_publication_time`). Modul `research` adaugă variante cu lag presupus
  într-un fișier separat, marcate „necertificat point-in-time”. Revizuirile sunt păstrate ca versiuni.
* Fără backward fill, ferestre centrate, interpolare temporală sau normalizare pe întreg datasetul; z-score-urile
  exclud valoarea curentă; seriile zilnice sunt procesate pe seria zilnică originală și apoi îmbinate.

## Barele

* Execuții individuale ordonate după `trade_id`; deduplicare după identitatea reală (duplicate identice eliminate și
  numărate, conflicte = eroare); prețuri/cantități ca întregi scalați exacți (fără float la praguri).
* Time bars `[t, t+Δ)` aliniate la epoca Unix; interval fără tranzacții cu feed complet = volum 0, OHLC null;
  interval atins de un gol = incomplet, OHLC null.
* Volume / notional bars: închidere la prima tranzacție care atinge sau depășește pragul, tranzacția inclusă integral,
  depășirea rămâne în bară, următoarea bară începe cu următoarea tranzacție; acumularea începe la `bar_origin` și nu se
  resetează la granițe de fișiere/zile/batch-uri. Notionalul este în moneda de cotare (USDT ≠ USD).
* Gol în ID-uri: încercare de recuperare prin REST (`historicalTrades`); dacă eșuează, bara deschisă e invalidată și
  începe un segment nou, documentat în manifest.
* Rezultatele sunt identice bit cu bit la procesare integrală, pe batch-uri sau după restart (testat).

## Limitări importante

* Nimic din istoric nu a fost descărcat în mediul de construcție (rețea blocată). Acoperirea istorică reală a fiecărei
  surse trebuie confirmată cu `dry-run`/`download` pe o mașină cu acces.
* Formulele AFML sunt implementate după specificația sarcinii și verificate numeric față de referințe; **textul
  cărții nu a fost accesibil** (vezi [`docs/FORMULAS.md`](docs/FORMULAS.md)); comutatorul
  `features.structural.formula_gate=primary_verified_only` le face null.
* L2, lichidări, skew: fără istoric gratuit continuu ⇒ null în istoric; disponibile din colectarea live
  (și din zilele gratuite Tardis, eșantioane discontinue).
* Fluxul Binance `forceOrder` este eșantionat (ultima lichidare per 1000 ms) — coloanele se numesc
  `binance_reported_sampled_liq_*`, nu „total lichidări”. Bybit raportează prețul de faliment, nu de execuție.
* Spot `bookTicker` nu are timestamp de exchange: se folosește momentul recepției.
* Block trades Binance (din 2026-05) au flux separat; nu se știe dacă apar în arhiva `trades`; nu sunt incluse în bare.
* Neimplementat: bare USD-notional, fallback de time bars din klines native, variante HAC, frontiere critice CSW.
* Colectorul live salvează date brute; nu există un server de feature-uri în timp real (replay-ul reconstruiește din loguri).

## Structura codului

```
crypto_dataset.py          punct de intrare CLI
cryptods/config.py         configurație, validare, hash-uri, identitatea seriilor
cryptods/bars.py           kernel exact (numba) pentru time/volume/notional bars, stare, goluri
cryptods/sources/          Binance (arhivă + REST), Coinbase, Deribit, Alternative.me/Farside/Wayback, Tardis
cryptods/book/             carnet L2 exact (reguli de sincronizare Binance spot/futures, Bybit, Tardis), adaptoare
cryptods/features/         basic, flow (λ), structural (SADF/QADF/CADF/SDFC/CSW/BDE), crossmarket, external,
                           options (skew), liquidations, book_features (L2, cost, concentrări), quotes (spread)
cryptods/asof.py           îmbinări backward as-of, vintage-uri point-in-time
cryptods/live/             colector WebSocket/poll și replay
cryptods/pipeline.py       moduri, asamblare, manifest, acoperire
tests/                     teste unitare, de referință și anti-lookahead (inclusiv pe date reale)
docs/                      STATUS, FORMULAS, FEATURES (registru generat), CONFIG, SCHEMA, SOURCES
sample_output/             dataset real mic + manifest + calitate + acoperire
```
