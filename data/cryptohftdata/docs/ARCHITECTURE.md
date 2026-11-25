# Arhitectura `chdds`

## 1. Straturi (raport §8.4)

| Strat | Unde | Ce conține | Cum se produce |
|---|---|---|---|
| 1. Raw imuabil | `cryptohftdata_raw/binance_futures/YYYY-MM-DD/HH/BTCUSDT_<stream>.parquet[.zst]` | fișierele furnizorului, nemodificate | descărcare (scripturile existente) |
| 1b. Audit + staged | `derived/audit/per_file/…json`, `derived/staged/<stream>/<day>/<HH>.parquet` | statistici exhaustive per fișier (SHA-256, schemă, metadate, null-uri, ceasuri, duplicate, goluri, lanț L2) și copia exactă a rândurilor ca întregi scalați (preț×10², cantitate×10³, mark/funding/OI×10⁸) + flag-uri de parsare | `chdds audit` |
| 2. Evenimente canonice | `derived/canonical/` | trades economice deduplicate în ordinea disponibilității (`seq`, `seg`), goluri de ID; mark/ticker deduplicate; OI cu versiuni la prima recepție; lichidări cu amprentă conservatoare | `chdds canonical` |
| 3. Stări și primitivi | `derived/book_grid/…/grid100ms`, `derived/book_state/` | replay L2 cauzal (stare, domeniu certificat, integrale), grilă top-of-book la 100 ms, checkpoint-uri zilnice, statistici orare și jurnal de verificare contra snapshot-urilor | `chdds book` |
| 4. Bare și FeatureSnapshots | `derived/bars/<spec_id>/`, `derived/features/`, `derived/book_features/`, `derived/datasets/` | bare (complete/parțiale, stare de reluare), părți de features pe zile, dataset final pe luni + manifest | `chdds bars/features/book/assemble` |
| 5. Ținte și experimente | — | **intenționat absent** | etapă ulterioară |

Book-ul nu se copiază canonic (58,6 mld. rânduri): fiecare fișier orar este decodat, mesajele sunt reasamblate
(`book/messages.py`), lanțul de secvență este rezolvat cauzal (`book/chain.py`) și starea este aplicată de un kernel
numba (`book/core.py`). Un singur replay evaluează simultan momentele de decizie ale tuturor seturilor de bare.

## 2. Ceasuri

- **Ceasul principal = disponibilitatea** (`received_time`, ns). Barele de timp sunt intervale `[T_k, T_k+Δ)` pe acest ceas;
  ferestrele `W=(t−w, t]` sunt pe acest ceas; orice observație folosită are `available_time ≤ eval_time`.
- **Timpul de piață** (`trade_time`, `event_time`, `transaction_time`, `timestamp` OI) este păstrat separat:
  `m_open_ms/m_close_ms` pe bare, vârste „de piață” (`*_mkt_age_ms`) pentru prospețime, `oi_reference_ts_ns`.
- Pentru book, timpul de disponibilitate al unui mesaj = recepția mesajului care completează lanțul (un mesaj sosit
  înaintea predecesorului său este aplicat doar când predecesorul sosește). Snapshot-urile cu `received_time` sintetic
  (= `event_time·10⁶`) nu dovedesc disponibilitate: validitatea pornește după conectarea primului update + 1 s.
- `decision_time`: bare de timp `T_{k+1}`; bare de eveniment = recepția tranzacției de închidere. `eval_time`
  (as-of inclusiv pentru celelalte fluxuri): `T_{k+1} − 1 ns` (starea din stânga limitei) / recepția tranzacției.
  Tranzacțiile sunt limitate și de cursorul `last_seq` (nu intră tranzacții ulterioare cu același timestamp).
- Latența de procesare nu este adăugată implicit (0); poate fi simulată ulterior prin deplasarea `eval_time`.

## 3. Fluxul canonic de tranzacții (raport §3.3, §7.1)

Identitate `trade_id`; copii identice → prima recepție autentică; ID cu payload diferit → carantină și rupere de
segment; `NA` (preț = cantitate = 0) exclus din economie dar folosit pentru continuitatea ID-urilor; ordine
canonică `(received_time, trade_id)`; `seq` global; un ID lipsă rupe segmentul (`seg`) și intervalul de recepție
dintre vecini devine gol de flux (`trades_gaps.parquet`).

## 4. Barele (raport §7)

- Tranzacția nu se fragmentează; tranzacția care atinge pragul închide bara; se emite cel mult o bară per tranzacție;
  excesul este `overshoot`, nu se reportează.
- Starea (bara deschisă, acumulatori CUSUM) trece peste fișiere/zile; checkpoint zilnic (`_state/<zi>.npz`), reluare
  incrementală. Bara încă deschisă la capătul intervalului construit este scrisă separat (`_partial_end_<end>.parquet`).
- Un gol de tranzacții închide bara de eveniment ca parțială (`close_reason=1`), resetează CUSUM și reia warm-up-ul;
  barele de timp care intersectează un gol au `feed_complete=0`.
- CUSUM simetric, drift 0, `h = log(1+b/10⁴)`; formulare exactă echivalentă `S+ = log(p/min p de la reset)`,
  comparată în întregi: `p·10⁶ ≥ p_min·(10⁶+b_c)`; egalitatea declanșează; implicit se resetează doar acumulatorul
  declanșat (`cusum_reset: both` = variantă distinctă). Bara conține **toate** tranzacțiile dintre frontiere,
  inclusiv cea declanșatoare. Test: echivalență exactă cu recursia float din raport.
- Praguri: fixe (valori rotunde, nealese pe date) sau programate zilnic exclusiv din zilele anterioare
  (volume/dollar: mediana volumului pe intervale de `target_s` din ultimele N zile; CUSUM: `c·σ` al randamentelor
  de 1 min din ultimele N zile). La schimbarea pragului CUSUM se închide bara ca parțială și se resetează acumulatorii.

## 5. Reconstrucția L2 (raport §3.4)

1. Reasamblarea mesajelor: rând = nivel; grup de recepție = `(event_type, final_update_id, received_time)`; mesaj =
   copiile aceluiași `(event_type, u)` (în 2025-06/07 fiecare mesaj apare de ~3 ori, intercalat la nivel de rând).
   Copia canonică = prima recepție nesintetică; copiile cu conținut diferit sunt numărate ca conflicte.
2. Resolver cauzal: update aplicabil numai dacă `pu == last_u`; sosirile timpurii sunt ținute în buffer și aplicate la
   recepția mesajului care completează lanțul; un gol necompletat în 60 s (ceas de recepție) este gol real → book
   neancorat până la un snapshot care se conectează (`U ≤ L ≤ u` sau `pu == L`). Un fișier orar lipsă este gol forțat.
   Cât e neancorat, rezolverul ține ultimele 60 s de mesaje, ca un client real, deci un snapshot se poate conecta la mesaje
   primite înaintea lui. Un snapshot sosit cu lanțul aparent intact și `L > last_u` rămâne în așteptare: dacă următorul mesaj
   nu continuă lanțul, dar se conectează la `L`, snapshot-ul devine ancoră (reconectare). Aplicarea se face la momentul
   recepției, niciodată mai devreme. Comportamentul vechi: `legacy_chain: true`.
   Duplicatele `u ≤ last_u` sunt verificate prin hash de conținut (identic → ignorat; diferit → invalidare).
3. Snapshot cu lanțul intact: verificare exactă când lanțul ajunge la `L` (stare = snapshot ⊕ primul mesaj conectat,
   nivelurile atinse de acel mesaj fiind excluse) în intersecția domeniilor certificate; în afara domeniului nostru
   valorile snapshot-ului sunt adoptate (extindere de domeniu). Nepotrivire → reconstrucție din snapshot (contorizată).
4. Domeniu certificat (exact, fără euristici): bid-urile sunt certificate pe `[F_bid, +∞)`, ask-urile pe `(−∞, F_ask]`.
   La ancoră `F_bid` = cel mai adânc bid al snapshot-ului, `F_ask` = cel mai adânc ask (deasupra best bid-ului unui
   snapshot complet nu există bid-uri; orice apariție ulterioară vine ca update). Extinderea folosește două fapte ale
   protocolului: (a) fluxul diff conține fiecare nivel modificat, deci un tick raportat după ancoră este cunoscut exact;
   (b) book-ul exchange-ului nu este încrucișat, deci orice ask la un preț ≤ un best bid certificat (și orice bid ≥ un best
   ask certificat) este 0 în acel moment și rămâne urmărit. Premisa e verificată exhaustiv pe flux: 1,05 mld. de mesaje, o
   singură excepție (2025-08-29 06:37–06:55, `docs/D1_REPORT.md` §6). Frontiera avansează numai peste tick-uri
   contigue cunoscute — un update izolat departe de mid nu certifică nivelurile dintre el și book-ul cunoscut. Benzi,
   top-K și costuri care ies din domeniu sunt `null`. Comparația cu snapshot-urile ulterioare verifică și zona extinsă.
   Grila de tick-uri acoperă 0–400.000 USDT; nivelurile din afara ei (checkpoint-urile furnizorului conțin ask-uri până
   la ~4,5 mil. USDT) sunt ignorate și nu extind domeniul (irelevante pentru benzile ≤ 50 bps).
5. Aplicare atomică pe mesaj complet; ștergerea unui nivel absent este permisă; book încrucișat → invalid; încrucișare
   > 1 s → ancora este invalidată. OFI/adăugări/reduceri nu traversează ancore sau goluri.
6. Validitate (`book_state`): 0 valid; 1 neancorat; 2 vechi (> 1 s vârstă de piață); 3 încrucișat; 4 ancoră
   neconectată; 5 warm-up după ancoră sintetică; 6 parte goală; 7 top în afara domeniului certificat;
   8 `healed_unverified` (numai în modul experimental `heal`, niciodată în dataseturile implicite);
   9 și 10 numai în `book_mode: rebuilt` (mai jos).
7. Optimizare exactă: în modul strict, orele fără niciun snapshot în care resolver-ul este neancorat nu sunt decodate
   (starea nu poate redeveni validă fără snapshot); rezultat identic (testat).

### Modul opțional `book_mode: rebuilt`

Fără ancoră, book-ul pornește gol la primul mesaj și după fiecare gol real, de la cel mai vechi mesaj din buffer. Regula de 60 s
e aceeași, dar un gol se măsoară de când a devenit observabil, deci o cascadă costă o singură așteptare.

Un book pornit gol e submulțime a book-ului real: tot ce ține a fost raportat. Cu `b, a` cele mai bune niveluri proprii,
`b ≤ b* < a* ≤ a`, deci un bid ascuns poate sta doar strict între ele. Dacă toate tick-urile dintre `b` și `a` au fost raportate
pe partea bid, `b` e exact (simetric pentru ask; la spread de un tick nu e nimic între). De aici frontiera avansează ca la pct. 4.
`book_state` 9 = vârf certificat fără ancoră; 10 = vârf necertificat, features `null`. Un snapshot conectat readuce starea 0.

Numai în acest mod: rangul k din MLOFI cere rangul în domeniu toată fereastra; adăugările/reducerile cer banda de 25 bps
certificată toată fereastra; minimele pe bandă cer banda în domeniu toată fereastra; ferestrele cer starea validă cu 1 ms înainte
de marginea stângă. În ambele moduri: markouts și effective spread cer în plus `h` și vârsta maximă a cotației, iar scadențele de
markout nu trec peste sfârșitul lotului. Numai în modul reconstruit: după un mesaj încrucișat certificatul se anulează, nu se emite nimic 600 s (`rebuilt_cross_quarantine_s`), iar la
final book-ul repornește gol, deci certificarea se recâștigă doar din raportări noi. Regulile de markout se aplică în ambele moduri.

Integralele temporale sunt exacte pe segmente constante între evenimente; marginea stângă a ferestrei folosește
starea în vigoare la `t−w` (ring buffer de integrale cumulative + rata în vigoare, limitele ca offset-uri întregi);
sumele cumulative sunt rebazate la fiecare oră (precizie float64 controlată). Coverage strict = 1 (toleranță 1e−7).

## 6. Features

Primitivii se calculează o singură dată pe set de interogări: tranzacțiile prin sume prefix exacte în întregi
(volum ×10³, notional ×10⁵) + segment tree (range) + grile regulate (RV); fluxurile lente prin as-of pe recepție;
book prin replay. Registrul (`chdds/registry.py`) mapează fiecare coloană la o definiție; testul de acoperire
verifică că nu există coloane neînregistrate sau ambigue. Selecția coloanelor pentru un experiment se face prin
`features.select` (globs), separat de prioritatea de implementare.
