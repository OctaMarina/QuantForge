TITLE = r"""
# Diagnostic predictiv multi-activ — dataseturile `crypto_dataset_pipeline` (BTC, ETH, SOL, XRP, BNB, 2021–2026)

**Întrebarea:** conțin dataseturile informație despre randamentele viitoare (15 min, 2 h, 1 zi) peste un reper simplu,
care familii de features o aduc și merită continuată cercetarea?

Este un **prim diagnostic**, nu o strategie. Configurațiile (dataset, orizonturi, seturi de informație, modele, folduri,
criterii) au fost fixate înainte de a vedea vreo performanță în afara antrenării; grila exactă este în `run_grid.py`.
Codul este în `mad_lib.py` (același director); calculele grele sunt în cache (`~/.cache/quantforge/multiasset_diag/v1`).
Datele originale sunt doar citite.
"""

SUMMARY = r"""
## Rezumat executiv

**Date.** 5 active (BTC, ETH, SOL, XRP, BNB), Binance spot, 2021-09 → 2026-09, numai bare *volume* / *quote-notional*
la ~25/50/100 bare pe zi; fără time bars și **fără ținte**. Book-ul L2, lichidările, ETF-urile, Fear & Greed și skew-ul
sunt nule în istoric, iar DVOL acoperă ~7%. „Premium” înseamnă (1) prima Coinbase USD față de Binance USDT pentru BTC, ETH
și activul propriu și (2) basis-ul, premium index-ul și funding-ul perpetualului propriu. Informația din alte active =
premium-urile BTC/ETH plus randamentele celorlalte active, pe care le-am construit la momentul deciziei.
**Nicio problemă nu invalidează experimentul** (secțiunea 2).

**Design.** Bare notional ~100/zi; orizonturi de 15 min, 2 h și 1 zi în timp real; ținta e randamentul log de la t+1 s
(control bounce de la t+60 s); un model comun ridge + LightGBM; 6 folduri de câte 6 luni (2023–2025), cu aceleași
granițe pentru toate activele și cu purjare. Perioada 2026-01 → 2026-08 este rezervată, dar nu e neatinsă pentru BTC.

| | rezultat |
|---|---|
| **Robust statistic** | **PV la 15 min**: R² OOS 0,84% (ridge, CI 0,40–1,29) / 2,0% (LightGBM, CI 0,94–2,80), pozitiv în 6/6 folduri și la toate cele 5 active; trece Holm, eliminarea celor 10 zile dominante (0,62 / 0,92%) și controlul bounce (0,70 / 1,68%). **Dar pe unitate de risc e minuscul** (R²σ 0,06–0,09%, Spearman 0,03–0,04), iar 56–72% din câștig vine din 10 zile de crash. |
| **Indicii fragile** | PV la 2 h (R² 0,7–1,2%, dar R²σ ≈ 0 și Holm p = 0,27 / 0,62); la 1 zi, randamentele cross-asset (+0,89 pp, p = 0,09) și structuralele (+0,66 pp) în LightGBM; derivatele pentru BNB la 15 min. |
| **Fără avantaj demonstrat** | **Toate familiile suplimentare** (flux, lambda, structurale, derivate, premium Coinbase, cross-asset) la toate orizonturile: 0 din 42 de teste incrementale trec Holm; `OWN` și `ALL` nu bat `PV`. Orizontul de **1 zi** în general (3/6 folduri, R²σ negativ). Premium-ul Coinbase e folosit intens de LightGBM la 1 zi, dar nu aduce câștig în afara antrenării. |
| **Economic** | Decila extremă la 15 min: ~10,5 bps brut per semnal (CI 4–16), aproape numai pe partea long; ~0 net la 10 bps cost dus-întors, negativ la 20 bps. Concentrat în panici, când execuția la ultimul preț de tranzacție e nerealistă. **Nu este o strategie demonstrată.** |

**Mecanism (explorator).** Semnalul PV de 15 min înseamnă în principal „după activitate și volatilitate extreme urmează
randamente pozitive”. Componenta stabilă de la an la an, inclusiv în 2022, e **revenirea după o scădere bruscă într-o stare
de volatilitate mare**. Restul depinde de regim: în 2022 dispare.

**Recomandare: un singur experiment.** Un studiu de eveniment confirmator, definit dinainte, pentru „revenirea după
capitulare” la 15 min, cu intrare la t+60 s, evenimente declusterizate și costuri conservatoare. Se evaluează **o singură
dată** pe perioada rezervată 2026-01 → 2026-08 și pe date de după 2026-09-01. Dacă eșuează, linia se oprește. Familiile
suplimentare și orizontul de 1 zi **nu** merită continuate în forma testată.
"""

S1 = r"""
## 1. Harta datelor

**Ce există.** 43 de dataseturi cu manifest în `data/datasets` (plus un director gol), fiecare cu **un singur tip de bară**: bare de volum (prag în moneda
de bază) și bare *quote-notional* (prag în USDT), calibrate la ~25 / 50 / 100 de bare pe zi, pentru BTC, ETH, SOL, XRP,
BNB (Binance spot, `trades` individuale), perioada **2021-09-01 → 2026-09-01**. **Nu există time bars** și **nu există
ținte** (pipeline-ul declară explicit că nu generează randamente viitoare). Coloanele `ret_bars_k` sunt randamente
*trecute* pe k bare, nu ținte.
"""

S1_CHOSEN = r"""
**Configurația aleasă (înainte de rezultate): bare notional, ~100 bare/zi, pentru toate cele 5 active.**

* *notional* și nu *volume*: prețurile s-au mișcat de 3–10× în perioadă, deci un prag în unități de bază schimbă
  enorm frecvența în timp; pragul în USDT e mai stabil (deși tot variază, vezi tabelul de mai jos);
* *100 bare/zi*: cele mai multe puncte de decizie, iar durata mediană a unei bare (~8–10 min) se potrivește cu
  orizontul scurt; frecvența de bară schimbă doar momentele de decizie și ferestrele în bare, nu orizontul în timp real;
* ETH există numai în build-ul vechi (cod `6a1d922b`); diferențele față de codul curent sunt doar denumirea
  directorului, refactorizarea listei de active premium și o verificare de overflow `int64` adăugată ulterior.
  Pentru ETH scala este 10^10 și un prag de 12 mil. USDT, de ~770× sub limita de overflow — build-ul e valid.

Fiecare rând = o bară închisă; decizia se ia la `decision_time = bar_end` (latență declarată 0). Când mai multe bare se
închid pe aceeași tranzacție (același timestamp) se păstrează ultima, ca să nu existe decizii duplicate.
"""

S1_BARS = r"""
**Ce ar însemna o țintă „în bare”.** Datasetul nu are ținte, dar dacă am fi folosit „următoarele k bare”, durata lor
efectivă ar fi variat enorm între active și ani (mediana în minute, mai jos) și mai ales în interiorul aceleiași zile
(p10–p90 al unei bare: de la ~1 minut la peste o oră). De aceea toate țintele de aici sunt **orizonturi în timp real**.
"""

S1_FAM = r"""
**Familiile de features și acoperirea lor** (media pe luni a ponderii valorilor valide, pe datasetul ales). Familiile
L2 / cost / quotes, lichidări, ETF, Fear & Greed și skew sunt practic nule în istoric (necesită colectare live sau
dovadă point-in-time; în modul `strict` sunt null). DVOL acoperă < 50% și discontinuu — exclus. Rămân utilizabile:
randamente, volatilitate, durată/intensitate, flux agresor, lambda Kyle/Hasbrouck, structurale (SADF/QADF/CADF/SDFC/
CSW/BDE), derivate (basis perp–spot, mark/index, premium index Binance, funding), trend 30 zile și premium Coinbase.
"""

S1_PREMIUM = r"""
**Ce înseamnă „premium” și ce vine de la alte active.** Există două familii diferite:

1. **Premium Coinbase (USD vs USDT)** — `premium_raw_coinbase_<x>_1m = log(P_Coinbase <x>-USD / P_Binance <x>USDT)` pe
   același minut închis; `premium_fx_*` îl corectează cu cursul Coinbase USDT-USD (deci `raw − fx` = abaterea USDT de la
   1 USD); `_1d` = media zilei UTC precedente (acoperire ≥ 90%), `_1d_z90` = z-score cauzal pe 90 de zile anterioare.
   Se calculează pentru **BTC și ETH în toate dataseturile (active de context)** și pentru activul propriu. Pentru XRP
   începe în iulie 2023 (relistare Coinbase); pentru BNB acoperirea e ~10% — practic absent.
2. **Premium de derivate al activului propriu** — basis perp–spot Binance USD-M (log), log(mark/index), Premium Index
   publicat de Binance și ultimul funding realizat.

**Informația din alte active în dataseturi** se reduce la premium-urile Coinbase ale BTC/ETH (plus DVOL BTC/ETH,
parțial, și ETF BTC/ETH, nul). Pentru setul cross-asset am construit în plus **randamentele celorlalte 4 active la
momentul deciziei** (5 min, 1 h, 4 h, 1 zi) direct din grila de preț pe secundă a fiecăruia (ultima tranzacție
*strict înainte* de decizie), normalizate cu volatilitatea zilnică a acelui activ cunoscută la acel moment: media
celorlalte active (`xmkt_*`), activul-lider (`xlead_*` = BTC; pentru BTC, ETH) și randamentul relativ (`rel_*`).
Toate activele au același ceas (timestamp de exchange Binance), deci alinierea e exactă.
"""

S2 = r"""
## 2. Verificări care ar putea invalida predicția

Folosim manifestele și testele existente (80 de teste, inclusiv anti-lookahead) și refacem doar verificările ieftine
care leagă datele de țintele noastre:
"""

S2_VERDICT = r"""
**Verdict: nicio problemă care să invalideze experimentul.**

* un singur segment pe serie, fără goluri de ID, fără zile lipsă; barele de 1 min reconstruite coincid cu klines
  Binance (≤ 2 nepotriviri din 2,67 mil. minute); ~72 de minute există doar în klines (opriri ale exchange-ului) —
  țintele care ar folosi un preț mai vechi de 120 s sunt invalidate;
* prețul folosit pentru ținte (grila px1s, ultima tranzacție strict înainte de un moment) reproduce **exact**
  `ret_5min` din dataset → aceeași convenție de timp ca la features;
* premium-ul de 1 minut la decizie corespunde în 100% din cazuri minutului **închis** înainte de decizie (0% pentru
  minutul deschis), iar corelația dintre randamentele de 1 min Coinbase și Binance are vârful la lag 0 (~0,99): nu
  există decalaj de etichetare a lumânărilor Coinbase. Rămâne o limită: în market-time replay minutul Coinbase e
  considerat disponibil exact la sfârșitul lui; live ar întârzia câteva secunde — neglijabil pentru orizonturi ≥ 15 min;
* îmbinările externe sunt numai backward as-of pe `available_at` cu egalitatea exclusă; seriile zilnice (premium 1d,
  trend 30z) devin disponibile la sfârșitul zilei UTC; z-score-urile exclud valoarea curentă;
* **limite reale (nu invalidante):** (i) pragurile barelor au fost calibrate pe volumul mediu al întregii perioade —
  privire înainte doar în *rata de eșantionare*, nu în direcția prețului; (ii) prețurile sunt de tranzacție, nu mid:
  pasul de preț ajunge la ~6,6 bps (SOL 2023) și ~4 bps (BNB 2023), deci bounce-ul bid/ask poate crea o
  „predictibilitate” mecanică la orizontul scurt. De aceea am declarat dinainte un control cu intrare la t+60 s;
  (iii) deciziile sunt pe ceas de evenimente, deci perioadele active cântăresc mai mult (2026: ~50 bare/zi față de
  100–160 în 2024–2025).
"""

S3 = r"""
## 3. Designul experimentului (fixat înainte de rezultate)

**Orizonturi în timp real** — alese din structura datelor: durata mediană a unei bare la 100/zi este ~8–10 min, iar
features-urile au ferestre de la o bară la 30 de zile (randamente 5 min…30 z, premium zilnic, funding la 8 h, trend 30 z):

| orizont | rol | ce ar trebui să conteze |
|---|---|---|
| **15 min** | scurt (~1–2 bare) | flux, informația ultimei bare, lead–lag între active |
| **2 h** | intermediar (~10–15 bare) | momentum/reversal intraday, regim de volatilitate |
| **1 zi** | lung | premium zilnic, funding, trend, momentum pe zile |

**Ținta principală:** `y_h = log P(t+h) − log P(t+1s)`, cu P = ultima tranzacție strict înaintea momentului (grila px1s,
vechime ≤ 120 s). Intrarea la t+1s modelează o execuție imediată după închiderea barei. **Control bounce (15 min):**
`log P(t+15m) − log P(t+60s)` — un câștig la 15 min este considerat real numai dacă supraviețuiește acestui control.
Limite: prețuri de tranzacție (nu mid), deci la 15 min zgomotul bid/ask e de ordinul 1–3 bps pentru SOL/XRP/BNB
în 2022–2023; la 2 h și 1 zi e neglijabil față de volatilitate.

**Seturi de informație (cumulative + o familie pe rând peste PV):**
* `PV` — preț și volum: randamente trecute (1/5/20 bare; 5 min…30 z) normalizate cu volatilitatea, volatilitate realizată
  și rapoarte, durata barei / nr. tranzacții / mărimea medie relative la ultimele 200 de bare, range, poziția close-ului
  în bară, trend 30 z;
* `OWN` = PV + flux agresor + lambda + structurale + derivate + premium Coinbase propriu (*features suplimentare ale activului*);
* `ALL` = OWN + premium Coinbase BTC/ETH (când nu e activul propriu) + randamentele celorlalte active (*informație cross-asset*);
* `PV+FLOW`, `PV+STRUCT`, `PV+DERIV`, `PV+CBP` (premium Coinbase, toate activele), `PV+XRET` — comparații între familii.

**Modele:** ridge (închis, α ales pe validarea internă dintr-o grilă de 7 valori) și LightGBM (15 frunze, ≥ 2000
observații pe frunză, rată 0,03, număr de arbori ales prin early stopping pe validarea internă, apoi refit), plus
reperul **media istorică**. **Un model comun pentru cele 5 active** — features-urile sunt construite identic și fără
unități (randamente normalizate, rapoarte, t-statistici), iar semnalul așteptat e slab, deci gruparea reduce varianța.
Ținta de antrenare este scalată cu volatilitatea ex-ante (`rvol_1d·√(h/1z)`) și tăiată la ±6σ; predicția se
re-exprimă în bps. Scalarea robustă a features-urilor e învățată **per activ, numai pe antrenare**.

**Validare:** walk-forward cu fereastră extinsă, **aceleași granițe pentru toate activele**, 6 perioade de test de
câte 6 luni (2023H1 … 2025H2), antrenare de la 2021-09-08. **Purjare:** un rând de antrenare intră numai dacă eticheta lui
se termină înainte de începutul testului (t + h + 1 s < start test); la fel pentru validarea internă (ultimele 6 luni
ale ferestrei). Nicio transformare sau selecție nu vede testul.

**Perioada rezervată:** **2026-01-01 → 2026-09-01** nu este evaluată în acest notebook (țintele ei nici nu sunt
calculate). Atenție: **nu este neatinsă pentru BTC** — studiile anterioare pe datasetul `btcusdt_v2` (alt pipeline, alte
features, orizonturi de secunde–60 min) au folosit iul. 2025–oct. 2026. Pentru o confirmare cu adevărat curată sunt
necesare date de după 2026-09-01 (pot fi construite cu `update`). Toată perioada 2023–2025 de aici este explorată.

**Raportare și comparații multiple:** R² OOS față de zero (randamente brute, bps) și față de media istorică; R² scalat
la volatilitate (fiecare decizie cântărește la fel în unități de risc); corelații Pearson/Spearman; randamente pe
chintile de predicție; R² pe fold și activ; R² fără cele 10 zile dominante. Incertitudine: **bootstrap pe blocuri de
20 de zile** (respectă suprapunerea etichetelor și dependența). Familia principală de teste (pooled, 48 de teste) este
corectată **Holm**; testele per activ (210) sunt descriptive, cu BH. **Criteriu de robustețe declarat:** câștig pozitiv
pe scara brută *și* scalată, după eliminarea celor 10 zile dominante, în ≥ 5/6 folduri și la ≥ 4/5 active.
"""

S4 = r"""
## 4. Rezultate: tabel comparabil pe active, orizonturi și seturi de informație
"""

S5 = r"""
## 5. Ce familii aduc informație suplimentară și se schimbă rolul lor cu orizontul?
"""

S6 = r"""
## 6. Stabilitate, episoade extreme și controlul bounce
"""

S7 = r"""
## 7. Relevanță economică (ipotetică)

Fără costuri reale, arătăm randamentul realizat pe chintile de predicție și o regulă minimă de semn, cu sensibilitate
la costuri dus-întors de 2 / 5 / 10 / 20 bps (pentru referință: taxa taker Binance spot standard este 10 bps pe parte,
deci ~20 bps dus-întors fără reduceri; un maker cu rabat poate fi aproape de 0, dar cu risc de neexecutare).
Pozițiile se suprapun, deci cifrele sunt medii per semnal, **nu** rezultatul unei strategii.
"""

S8 = r"""
## 8. Ce este, de fapt, semnalul PV de 15 minute? (explorator)

Coeficienții ridge (în medie pe folduri) cu semn pozitiv: numărul de tranzacții și mărimea medie a tranzacției relative
la ultimele 200 de bare (+0,077 / +0,072), creșterea vol 1h/1z, `lrv_1d`; cu semn negativ: durata relativă a barei
(barele rapide → randament pozitiv ulterior), `lrv_1h`, randamentul pe 5 min (reversal). Pe scurt: **după bare foarte
rapide, cu volum mare, în volatilitate mare — în special după scăderi — prețul tinde să crească în următoarele minute.**
Tabelele de mai jos arată că regula câștigă aproape numai pe partea long și că o parte din efect depinde de regimul de
piață al perioadei de test (2023–2025, predominant crescătoare).
"""

CONCL = r"""
## 9. Concluzii

### 9.1 Harta datelor și limitele relevante
* **43 de dataseturi** (cu manifest), câte un tip de bară per director: volume / notional × ~25/50/100 bare/zi × 5
  active, plus build-uri vechi BTC de calibrare. ETH există doar în codul vechi, dar e echivalent. Fără time bars, fără ținte.
* **Familii utilizabile:** randamente, volatilitate, durată/intensitate, flux agresor, lambda Kyle/Hasbrouck,
  structurale, derivate (basis/premium index/funding), trend 30 z, premium Coinbase BTC/ETH (+ SOL; XRP din iulie 2023;
  BNB ~10%). **Nule sau aproape nule:** L2/cost/quotes, lichidări, ETF, Fear & Greed, skew; DVOL ~7%.
* **Limite reale:** (i) prețuri de tranzacție, nu mid; bounce-ul contează la 15 min pentru SOL/XRP/BNB în 2022–2023
  (pas de preț de până la 6,6 bps) și e tratat prin controlul t+60 s; (ii) eșantionare pe ceas de evenimente, cu pondere
  mai mare pentru perioadele active; (iii) praguri de bară calibrate pe toată perioada, cu efect numai asupra frecvenței;
  (iv) market-time replay: disponibilitatea minutului Coinbase e cea teoretică.

### 9.2 Tabelul comparabil (secțiunea 4) — citire
* **15 min:** toate seturile au R² brut pozitiv la toate activele (ridge 0,48–1,1%, LightGBM 0,3–3,3%). Diferențele dintre
  seturi sunt însă mai mici decât incertitudinea, iar `PV` e cel mai bun sau la egalitate. Pe scara scalată la risc,
  R² e de ordinul 0,03–0,46% pentru BTC/SOL, ~0 pentru ETH/XRP și negativ pentru BNB în LightGBM.
* **2 h:** R² brut pozitiv în medie (0,4–1,3% pooled), dar câștigul vine aproape integral din 10 zile (pondere
  0,7–1,0), iar R²σ ≈ 0; BTC, ETH și SOL sunt instabile între folduri.
* **1 zi:** semn instabil (BTC negativ în toate seturile, SOL în aproape toate), 1–4/6 folduri pozitive, R²σ negativ în
  majoritatea configurațiilor. R² brut pozitiv pentru ETH/XRP provine dintr-o pariere implicită pe trend (fold 0, 2023H1,
  e puternic negativ pentru toți), nu din informație stabilă.
* Media istorică (reper) are R² apropiat de 0 (0–0,2% la 15 min, negativ la 1 zi); modelele o bat la 15 min, nu și în mod fiabil la 1 zi.

### 9.3 Familii și orizont
* **Nicio familie nu aduce informație suplimentară demonstrată peste `PV`.** 0/42 de teste pooled trec Holm; 4/210
  comparații per activ au q_BH < 0,10, toate cu efect ≤ 0,12 pp după eliminarea a 10 zile.
* **Fluxul agresor / lambda** nu adaugă nimic la 15 min (LightGBM −0,30 pp, 1/6 folduri mai bune). Informația din fluxul
  barelor de eveniment, găsită anterior la secunde, nu supraviețuiește până la 15 min cu intrare la t+1 s.
* **Rolul se schimbă cu orizontul în ce folosește modelul, nu în ce câștigă în afara antrenării:** în LightGBM `ALL`,
  ponderea structuralelor scade de la 17% (15 min) la 0% (1 zi), cea a premium-ului Coinbase crește de la 13% la 45%,
  iar randamentele cross-asset ating maximul la 2 h (30%). Câștigurile OOS nu urmează însă importanța: premium-ul are
  ΔR² nesemnificativ și de semn instabil la 1 zi (−0,17 pp LightGBM / +0,09 pp ridge), iar cross-asset are semn negativ la 15 min / 2 h și pozitiv (fragil) doar la 1 zi.

### 9.4 Robust, fragil, fără avantaj
* **Robust (statistic, nu economic):** predictibilitate mică la **15 min din preț și volum**, la toate activele și în
  toate perioadele, care supraviețuiește corecției Holm, eliminării zilelor extreme și controlului bounce. Mărimea pe
  unitate de risc e foarte mică și o mare parte vine din câteva episoade de crash (10 octombrie 2025, 3 februarie 2025,
  decembrie 2024).
* **Fragil:** PV la 2 h; la 1 zi, randamentele cross-asset și structuralele în LightGBM; derivatele pentru BNB la 15 min;
  orice câștig economic la 2 h (un fold la −44 bps, susținut de partea long în volatilitate mare).
* **Fără avantaj demonstrat:** flux, lambda, structurale, derivate, premium Coinbase (propriu și BTC/ETH) și randamentele
  altor active, la toate cele trei orizonturi; seturile cumulative `OWN` și `ALL`; orizontul de 1 zi; câștigul economic
  net după costuri realiste de taker (≥ 15–20 bps dus-întors).

### 9.5 Un singur experiment următor
**Studiu de eveniment confirmator „revenire după capitulare” la 15 min, definit acum și rulat o singură dată pe date
nefolosite.**
* *De ce acesta:* este singura componentă care (a) apare în afara antrenării la toate activele și perioadele, (b) rezistă
  controlului bounce și (c) e prezentă descriptiv și în 2022, an de scădere. Componenta „volatilitate mare → randament
  pozitiv” depinde de regim și nu merită urmărită separat.
* *Definiție fixată înainte:* eveniment = bară închisă cu `zr_5m < −2` în timp ce `lrv_1h` e peste percentila 90 a
  propriilor ultime 90 de zile (prag cauzal); un singur eveniment per activ la 60 de minute (declusterizare); intrare la
  t+60 s; ieșire la 15 min (secundar 2 h). Repere: momente aleatoare în aceeași stare de volatilitate, fără scădere, și
  randamentul necondiționat.
* *Date:* perioada rezervată 2026-01-01 → 2026-09-01 (curată pentru ETH/SOL/XRP/BNB; pentru BTC văzută anterior în alt
  context) **și** date de după 2026-09-01, construite cu `update`.
* *Criteriu de succes, stabilit dinainte:* media declusterizată netă de 20 bps dus-întors > 0, cu CI bootstrap pe blocuri
  care exclude 0 la nivel pooled, și semn pozitiv la ≥ 4/5 active. Dacă nu e îndeplinit, **linia se oprește.**
  Execuția reală în panică (spread-uri lărgite) rămâne neverificată fără cotații bid/ask. O confirmare statistică ar
  justifica abia apoi colectarea de cotații, nu o strategie.
* **Nu recomand** continuarea familiilor suplimentare, a orizontului de 1 zi sau o căutare de hiperparametri pe
  configurațiile de aici.
"""
