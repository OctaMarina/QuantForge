# Raport de audit — CryptoHFTData BTCUSDT perpetual (Binance USDⓈ-M), 2025-06-28 → 2026-10-31

Data: 2026-11-03. Tabelele lunare complete sunt generate automat în `docs/AUDIT_TABLES.md` și CSV-urile din
`derived/audit/`; harta de validitate în `docs/VALIDITY_MAP.md`. Datele brute nu au fost modificate.

## 0. Ce este exhaustiv și ce este eșantion

| Verificare | Acoperire |
|---|---|
| Inventar (manifeste ↔ fișiere ↔ grilă orară), SHA-256, schemă, metadate, writer, număr de rânduri citite ↔ footer | **EXHAUSTIV**: toate cele 70.354 fișiere disponibile (= statusul `ok` din `rest_manifest.csv`), 60.511.118.623 rânduri citite |
| Null-uri, scale zecimale (parsare exactă, 0 valori inexacte/malformate), tick, valori nepozitive, tipuri, ceasuri, întârzieri, regresii, goluri de recepție intra-fișier | **EXHAUSTIV**, pe fiecare rând |
| Copii/duplicate/conflicte în fișier și între fișiere (trades, mark, ticker, OI, lichidări) | **EXHAUSTIV** (stratul canonic) |
| Book: reasamblarea mesajelor, copii, lanțul `pu→u` în ordinea recepției, snapshot-uri/checkpoint-uri, adâncime | **EXHAUSTIV** per fișier; continuitatea cauzală cu buffering și reconstrucția — prin replay-ul complet (secțiunea 9) |
| Validitatea pe familii (grilă de 1 s pentru fluxurile lente/trades; grilă de 100 ms pentru book) | **EXHAUSTIV** pe tot istoricul |
| Verificarea reconstrucției contra snapshot-urilor exchange-ului | toate snapshot-urile întâlnite cu lanțul intact (replay complet) |
| Experimentul „heal” (book continuat peste goluri fără ancoră) | **EȘANTION**: 2026-05-14 → 16 |
| Benchmark-uri | **EȘANTION**: 2026-04-22 → 25, 2026-05-14 → 16 + rularea completă |

## 1. Inventar și manifeste (exhaustiv)

- Grilă: 11.784 ore × 6 fluxuri = 70.704 chei; manifestul REST are 70.704 chei unice: 70.354 `ok`, 350 `unavailable`
  (lista `rest_unavailable_or_errors.csv` coincide exact cu subsetul). Toate fișierele `ok` există pe disc cu
  dimensiunea din manifest; niciun fișier pe disc fără `ok`; niciun fișier `ok` cu zero rânduri.
- Formate: 59.769 fișiere `.parquet.zst` (zstd exterior, Parquet necomprimat, descărcare S3) și 10.585 fișiere
  `.parquet` (zstd intern, descărcare REST începând cu 2026-08-19 10:00). Ambele sunt citite identic.
- `download_manifest.csv` (12.146 rânduri: 12.140 ok, 5 erori, 1 eroare de listare) nu recuperează nicio cheie
  indisponibilă în REST.
- Intervale indisponibile (capăt exclus): toate fluxurile 2025-06-28 00:00–05:00 (OI până la 06:00); book 4 întreruperi
  în 2026-03-30 → 2026-04-09 (201 h); mark și ticker 2026-04-23 17:00 → 04-24 17:00; lichidări 75 de ore (23 h pe
  04-23/24 + 37 ore izolate + câteva perechi). Ore cu toate cele 6 fișiere: 11.506 (97,64%).

## 2. Scheme, versiuni, regimuri ale colectorului

- O singură schemă per flux, cu o excepție: book 2025-06-28 05:00 → 2025-06-30 (54 fișiere) nu are coloana
  `order_count` (integral null după). Writer `parquet-rs 55.1.0` (OI: `parquet-cpp-arrow 25.0.1`).
- Metadate de reparație: `chd.repairs=["binance_futures_oi_cross_hour_dedup/v1"]` în 10.900 fișiere OI
  (2025-06-28 → 2026-09-25); 878 fișiere OI fără marcaj (inclusiv tot octombrie 2026). Marcajul nu este uniform și
  nu garantează deduplicarea: canonicul a eliminat 9.233 copii OI republicate în ore ulterioare.
- **Regimuri descoperite (nedocumentate de raportul inițial):**
  1. 2025-06-28 → 2025-06-30: fiecare eveniment este recepționat de ~3 ori (copii intercalate la nivel de rând, recepții
     diferite) în book (2,62 copii/mesaj), trades (5.411.452 rânduri-copie), mark și ticker. Tratament: deduplicare
     pe identitate (book `(tip,u)`, trades `trade_id`, mark/ticker `(event_time, payload)`), prima recepție autentică.
  2. Cadența mesajelor de book (mediană `event_time`): 102 ms până în iul. 2025 (`@depth@100ms`), 52 ms aug. 2025 →
     apr. 2026, 26–28 ms din mai 2026 → orice feature de „rată de mesaje” este întâi un control de feed.
  3. Snapshot-uri: iunie 2025 — 665 snapshot-uri REST la 5 minute, toate cu `received_time` sintetic
     (= `event_time·10⁶`); **iul. 2025 → 10.04.2026: zero snapshot-uri sau checkpoint-uri**; din 11.04.2026
     snapshot-uri REST reale după reconectări (~1–2/zi; 12.992 în mai, concentrate după 18.05); checkpoint-uri orare
     ale furnizorului din aug. 2026 (21), sept. (392), oct. (744 = fiecare oră).

## 3. Unități și ceasuri (exhaustiv)

- `received_time` ns, `event_time`/`trade_time`/`transaction_time`/`timestamp`/`next_funding_time` ms — confirmat pe
  toate rândurile (intervale plauzibile, parsare exactă). Prețuri: 2 zecimale, tick 0,1 respectat cu o excepție:
  133 rânduri de book (2025-07-10 21) la prețuri ~34–36k nealiniate la tick, toate cu cantitate 0 (ștergeri ale unor
  niveluri inexistente) → ignorate conform protocolului.
- Întârziere recepție−eveniment (mediane lunare): trades 124–143 ms, book 122–142 ms, mark 185–226 ms, lichidări
  123–143 ms, OI 336–401 s. Cozi: trades până la 244,6 s (iul. 2026), book până la 114,9 s, mark până la 568 s,
  lichidări până la 246 s → prospețimea se judecă pe vârsta de piață, nu pe vârsta recepției.
- Întârzieri negative (recepție înaintea evenimentului — decalaj de ceas al colectorului): trades 5.943 rânduri,
  book 1.687 mesaje, mark ~151 rânduri; păstrate, raportate.
- Recepții în afara orei fișierului: book 8 mesaje, trades 1, ticker 183 — toate *înaintea* orei (µs–ms; un mesaj de
  book cu 526 s). Prelucrarea pe ore le face disponibile cel mai devreme la începutul fișierului lor (conservator,
  fără lookahead).
- `trade_time ≠ event_time` la 432,6 mil. rânduri (diferențe de ms) — se folosește `trade_time` ca timp de piață.

## 4. Duplicate și conflicte

| Flux | Rezultat (exhaustiv) | Tratament |
|---|---|---|
| Trades | 5.411.452 copii identice (iun. 2025), 0 duplicate între fișiere, **0 conflicte** de payload pe același ID | prima recepție autentică |
| Book | 266.995.094 rânduri-copie eliminate, **0 copii cu conținut diferit**, 0 mesaje inconsistente, 0 niveluri duplicate într-un mesaj | prima recepție nesintetică; hash de conținut pentru duplicate ulterioare |
| Mark | 391.660 copii (iun. 2025), 0 conflicte | `(event_time, payload)` |
| Ticker | 364.182 copii, 1.745 `event_time` cu payload diferit (observații distincte, păstrate) | `(event_time, payload)` |
| OI | 9.233 copii republicate în ore ulterioare; **47 puncte de 5 min cu versiuni conflictuale** (D2) | versiuni la prima recepție; politica „prima versiune” + `oi_state=3` |
| Lichidări | 0 copii exacte, 0 copii cu altă recepție, 0 rânduri cu aceeași cheie (timp, sens, preț, cantitate) | amprentă conservatoare |

## 5. Trades: continuitate și rânduri speciale

- 1.780.200.969 rânduri; tipuri: MARKET 1.776.687.966, NA 3.482.637, INSURANCE_FUND 30.366. Toate rândurile NA și
  INSURANCE_FUND au preț = cantitate = 0 (niciun NA cu valori pozitive) → excluse din economie, păstrate pentru
  continuitatea ID-urilor. Tranzacții economice canonice: **1.771.277.674**.
- **2.846 goluri de ID** (12.799.476 ID-uri lipsă, ~0,72%), mediană 2.392 ID-uri/gol, durata pe ceasul de recepție
  mediană 2,4 s, maximum 980 s, total 9,2 h. Corespund reconectărilor (furnizorul confirmă golurile la reconectare);
  fiecare gol rupe segmentul de bare și invalidează ferestrele care îl intersectează.
- 2 inversiuni de ID în ordinea recepției; 0 recepții sintetice.

## 6. Mark / index / funding, ticker

- Mark ~1 rând/s după deduplicare; goluri pe ceasul de piață ≥30 s: 14–103/lună, dar 775 (aug. 2026) și 856
  (sept. 2026), dintre care 439 și 638 încep la minutul 01–05 al orei — confirmă nota furnizorului (hh:02–05 din
  mijlocul lui aug. 2026); revenire în oct. 2026 (27). Timp de gol 13,4 h (aug.) și 16,7 h (sept.).
- Întârzieri > 3 s la mark: până la 109.657 rânduri/lună (aug. 2026) → mark „rapid” (≤ 3 s) valid 94,2% în aug.
- Funding: rate între −0,0254% și +0,01% per perioadă; `next_funding_time` permite deducerea intervalului (nu se
  presupune 8 h).
- Ticker: fereastră statistică mobilă de ~24 h (verificată), relații preț/procent/VWAP controlate (vezi
  `ticker_monthly.csv`); ~1 rând/2 s; întrerupere 23–24.04.2026.

## 7. Open interest (exhaustiv)

- Puncte de 5 min (`timestamp`), loturi la ~490–616 s (mediană lunară; documentația curentă spune 5 min), întârziere
  minimă 1,4 s, mediană 336–401 s. La pornire: backfill de ~41 h (puncte cu `timestamp` până la 149.811 s înainte de
  recepție) — folosite numai după recepție.
- 141.821 puncte unice pe grilă după carantină; as-of după disponibilitate: punctul cu cel mai mare `timestamp` dintre
  versiunile recepționate până la `t`; un lot de puncte vechi nu devine o succesiune de actualizări.
- *(Actualizat 2026-11-05: tratamentul de mai jos e înlocuit; vezi `docs/LIMITATIONS.md` D2.)*
- **Anomalie 15–17.10.2026 (D2)**: 2.768 puncte cu timestamp la secundă (în afara grilei; două serii alternante,
  diferență ~350 BTC), recepționate 14:38–16:49 pentru 02:07–15:17 → carantină; 47 puncte de 5 min republicate pe
  17.10 cu diferențe de 1.035–8.678 BTC → prima versiune + `oi_state=3`. Procesarea afectată este oprită (valori
  `null`), restul OI continuă.

## 8. Lichidări (exhaustiv)

- 555.301 notificări, toate `FILLED`, `LIMIT`, `IOC`; `filled_quantity == quantity` la toate; `last_filled <
  filled` la ~22% (notificarea este starea cumulativă finală, nu o execuție parțială). BUY 254.622, SELL 300.679;
  notional notificat lunar 0,28–2,0 mld. USDT.
- Fără `order_id`: identitatea nu se poate demonstra; nu au apărut stări cumulative repetate ale aceleiași chei.
- Feed-ul public raportează cel mult o lichidare/simbol/1000 ms → sumele sunt „notional notificat”, nu total.
- Ore fără fișier: 75 → politica implicită `unknown` (nu zero). Un fișier prezent nu certifică feed-ul pe toată ora:
  se cere și continuitatea fluxului de piață comun (mark), altfel `liq_state=3`.

## 9. Reconstrucția book-ului

Procedura și regulile: `docs/ARCHITECTURE.md` §5. Rezultatele replay-ului complet sunt în `docs/VALIDITY_MAP.md`,
`derived/audit/book_replay_monthly.csv` și `derived/book_state/btcusdt_initial/` (statistici orare, jurnal de verificare,
contoare).

**Replay complet (EXHAUSTIV, varianta strictă):** 391.970.277 mesaje aplicate (16.428.344.962 niveluri), 5.942 ancorări,
630 invalidări (goluri reale), 0 book-uri încrucișate, 0 duplicate conflictuale; 8.227 de ore neancorate fără snapshot
nu au fost decodate (exact). **2.259 de verificări contra snapshot-urilor/checkpoint-urilor (657 iun. 2025, 488 mai,
21 aug., 368 sept., 725 oct. 2026) — 0 nepotriviri** în intersecția domeniilor certificate, inclusiv în zona extinsă
prin regula de frontieră; la cele 1.174 verificări cu `L` exact, topul book-ului coincide în 100% din cazuri.
Goluri reale pe lună: 4 (iun. 2025), 15–318 (apr.–mai 2026, maxim după 18.05), 27–49 (iun.–oct. 2026); goluri
rezolvate prin buffering (mesaje sosite înaintea predecesorului): 1.653 în mai, 418 în oct. 2026 (întârziere de
aplicare maximă 10,2 s). Validitate strictă a book-ului: 26,2% din timpul istoricului — iun. 2025 79%, iul. 2025 →
mar. 2026 0%, apr. 34%, mai 47%, iun. 51%, iul. 48%, aug. 58%, sept. 76%, oct. 2026 98%; topul în afara domeniului
certificat: 0% (regula de frontieră). În perioadele cu 1–2 snapshot-uri/zi, după fiecare gol real book-ul rămâne
neancorat până la următorul snapshot (cauza principală a celor ~50% invalide în apr.–aug. 2026).

- **Validare exactă**: pe 2025-06-28 → 30 reconstrucția pornită din prima ancoră coincide cu toate cele 657 de
  snapshot-uri REST ulterioare (0 niveluri diferite); pe tot istoricul 2.259/2.259. Două nepotriviri inițiale au fost explicate (niveluri din afara
  domeniului certificat al ancorei, nu erori) și regula de comparație a fost corectată.
- **Domeniul certificat**: snapshot-urile de 500 niveluri acoperă doar ~7–35 bps sub/peste mid; regula strictă
  „domeniul snapshot-ului” făcea topul book-ului necertificat după mișcări de ~0,2% (book valid 0–4% din timp în
  aprilie 2026). Regula adoptată — extinderea exactă prin invariantul „book neîncrucișat” și nivelurile raportate
  contigue — păstrează topul valid cât timp lanțul e continuu; benzile largi rămân necertificate unde nu sunt
  cunoscute (exact situația din raport). Comparațiile cu snapshot-urile verifică și zona extinsă.
- **Goluri de secvență**: în ordinea recepției apar rupturi multe (de ex. 566 în oct. 2025); o parte provin din
  conexiuni suprapuse (mesaje vechi sosite după cele noi) și sunt rezolvate prin buffering cauzal; restul sunt goluri
  reale (pauze ~2–3 s și ~60–70 s, backlog-uri de recepție de până la ~54 s urmate de salt de secvență).
- **Consecință (D1)**: după primul gol real din iulie 2025 nu există nicio ancoră până la 11.04.2026 → familiile
  dependente de book sunt `null` (`unanchored_book`) pe ~9,3 luni. Experimentul „heal” arată că o continuare fără
  ancoră devine rapid încrucișată → nu se aproximează.

## 10. Tratamente aplicate (situații cunoscute)

| Problemă | Tratament | Unde |
|---|---|---|
| Copii multiple ale aceluiași eveniment | prima recepție autentică; conflictele în carantină | `book/messages.py`, `canonical/*` |
| `received_time` sintetic la snapshot | nu probează disponibilitatea: validitate de la conectarea primului update + 1 s | `book/chain.py`, `core.py` |
| NA / INSURANCE_FUND zero | excluse din economie, păstrate pentru continuitate ID | `canonical/trades.py` |
| ID-uri lipsă | segment nou; bara deschisă închisă ca parțială; ferestre invalidate | `canonical/trades.py`, `bars/engines.py` |
| Mesaje sosite înaintea predecesorului | buffering; aplicare la recepția mesajului care completează lanțul | `book/chain.py` |
| Gol real / fișier lipsă | invalidare până la un snapshot care se conectează | `book/chain.py`, `driver.py` |
| Cantitate 0 la nivel absent; prețuri off-tick cu cantitate 0 | ignorate (protocol) | `book/core.py` |
| OI republicat | prima recepție a fiecărei versiuni | `canonical/slow.py` |
| OI în afara grilei de 5 min | carantină (D2) | `canonical/slow.py` |
| Fișier de lichidări lipsă | `unknown` (nu zero) | `features/slow.py` |
| Mark/ticker întârziat | prospețime pe vârsta de piață (3 s / 10 s) | `features/slow.py`, `core.py` |

## 11. Impactul pragurilor inițiale asupra acoperirii (măsurat pe tot istoricul, fără ajustare)

Fracția din timp în care familia este validă (`VALIDITY_MAP.md`): trades 99,88%; mark ≤ 3 s 98,60% (≤ 10 s 98,95%);
ticker ≤ 10 s 99,54%; OI ≤ 15 min 98,81%; lichidări (fișier + flux de piață continuu) 99,25%; book strict 26,22%.
Ferestrele stricte (coverage 100%) reduc disponibilitatea coloanelor cu fereastră lungă — bare de 1 min, tot istoricul
(`derived/audit/coverage_impact_time1m.csv`): ferestre de tranzacții 5 s 99,86%, 30 s 99,74%, 2 m 99,35%, 15 m
96,57%, 1 h 88,74%, 4 h 66,96%, 24 h 15,02%; lichidări 1 h 84,76%; normalizarea funding pe 7 zile 91,34%; mark
1 h 100% (coverage). La barele de 50 BTC (concentrate în perioade active) ferestrele de 24 h sunt valide 12%.
Disponibilitatea pe coloane (bare de 50 BTC, fără familia book): mediana 98,0%, decila inferioară 80,1%
(`feature_availability_volume50btc_nonbook.csv`). Pragurile nu au fost modificate; opțiunile sunt în `LIMITATIONS.md`
(D4).

## 12. Decizii încă necesare

Vezi `docs/LIMITATIONS.md`: D1 (book iul. 2025 → apr. 2026), D2 (OI 15–17.10.2026), D3 (ore fără fișier de
lichidări), D4 (variante parțiale pentru ferestrele lungi).
