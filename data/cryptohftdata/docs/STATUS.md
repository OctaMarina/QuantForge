# Stadiu: implementat / testat / executat pe istoricul complet

| Componentă | Implementat | Testat (unde) | Executat pe istoricul complet |
|---|---|---|---|
| Audit exhaustiv per fișier (schemă, SHA-256, null-uri, ceasuri, duplicate, goluri, lanț L2, snapshot-uri) | da | smoke + tabele | **da**: 70.354 fișiere, 60,51 mld. rânduri, 0 erori |
| Inventar ↔ manifeste | da | — | **da** |
| Strat staged (întregi zecimali exacți) | da | `test_decimal` | **da** (28 GB) |
| Trades canonice (dedup, NA, ordine disponibilitate, segmente) | da | `test_bars::test_real_canonical_conservation`, `test_features::*` | **da**: 1.771.277.674 tranzacții, 2.846 goluri |
| Mark/ticker/OI/lichidări canonice (+ carantina OI) | da | `test_oi_no_lookahead` | **da** |
| Harta de validitate (1 s / 100 ms) | da | — | **da** (`VALIDITY_MAP.md`) |
| Bare time/volume/dollar/CUSUM (fix + adaptiv cauzal), parțiale, reluare | da | 20 teste în `test_bars.py` | **da**: 9 seturi, conservare exactă verificată pe tot istoricul |
| Features F1–F3 (trades), F7 (mark/funding), F8 (OI), F9 (lichidări), F10 (ticker, calendar, sezonalitate) | da | `test_features.py` | **da** (9 seturi × 491 zile) |
| Resolver de lanț L2 cauzal + kernel de replay (F4, F5, F6-book, F7-mid) | da | 13 teste în `test_book.py` (inclusiv referință pur-Python, verificare contra snapshot-urilor, atomicitate, duplicate, gap, checkpoint, prefix) | **da**, varianta strictă; book valid doar în intervalele ancorate (D1) |
| Combinații F11 | da | registru | **da** (în asamblare) |
| Registru de features + inventar extensii | da | `test_registry.py` (inclusiv pe datasetul real) | toate cele 664–670 coloane mapate exact o dată |
| Varianta „heal” a book-ului | da (experimental, opțiune) | experiment 3 zile | **nu** (nerecomandată; D1) |
| Variante parțiale ale ferestrelor lungi (coverage < 100%) | **nu** | — | — (D4) |
| Extensii P1/P2 listate în `FEATURE_REGISTRY.md` („not_implemented”) | **nu** | — | — |
| Ținte / etichete / modele | **nu** (intenționat) | — | — |

Ce **nu** înseamnă rezultatele de mai sus: testele pe eșantioane (zilele 2025-06-28/29) nu validează întregul
dataset; verificările pe istoricul complet sunt cele marcate „executat”, iar verificarea independentă a datasetului
asamblat (`scripts/verify_dataset.py`) este făcută pe zilele listate în `derived/datasets/<nume>/verification.json`.

Execuții pe istoricul complet (2026-11-03/04): audit 70.354 fișiere; canonical; 9 seturi de bare (9.413.305 bare,
conservare exactă); features; replay L2 (2.259 verificări, 0 nepotriviri); asamblare (23 GB); verificare independentă
pe 5 zile × 9 seturi (0 erori); demo regenerat separat identic bit cu bit cu datasetul complet; 46 teste trec.

## D1 (2026-11-05)

| Componentă | Implementat | Testat | Executat |
|---|---|---|---|
| Corecții rezolver (buffer client, snapshot în așteptare, jumătăți de snapshot) | da | `test_book.py` | lanț pe tot istoricul; **dataseturile strict nu sunt regenerate** |
| `book_mode: rebuilt` (certificat exact, repornire după goluri, carantină), implicit în `btcusdt_v2` | da | `test_book.py`, inclusiv pe date reale | **da**, `btcusdt_v2` |
| Experiment R vs. book gol | da | contra implementării independente de vârste | 176 de intervale, 688 de starturi |
| Eșantion mare pe vârste + verificare de conținut | da | 357.880 de comparații cu book-urile explicite | toată referința (4.850 de ore), 3.388 de verificări |
| Scanare de stări încrucișate | da | — | **tot istoricul**, 491 de zile |
| Warm-up de timp ca mod de emisie | **nu** (măsurat, nerecomandat) | — | — |

## Pregătire regenerare (2026-11-05)

Vezi `docs/REGEN_READINESS.md`. Corecții implementate și testate. **`btcusdt_v2` e regenerat pe tot istoricul** (`docs/DATASET_V2.md`); `btcusdt_initial` rămâne neschimbat.

