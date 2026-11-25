# Information-driven bars: AFML §2.3.2 și variantele recomandate

Sursa definițiilor: *Advances in Financial Machine Learning* (López de Prado, Wiley 2018), §2.3.2, fragment
furnizat de utilizator. Toate cifrele de mai jos au fost **măsurate în acest proiect** (fixture real Binance
ETHUSDT spot 2020-08-14 10:00–15:00, 69 806 tranzacții, și date sintetice cu proprietăți cunoscute); nu sunt
preluate din altă parte.

**Pe scurt.** Librăria implementează definițiile AFML literal (implicit) și două variante **declarate, non-AFML**,
recomandate pentru dataseturi lungi:

| Familie | Regula AFML (literal) | Problema măsurată | Varianta recomandată |
|---|---|---|---|
| runs (TRB, VRB, DRB) | prag `E0[T]·max(P·E[v⁺], (1−P)·E[v⁻])`, `E0[T]` = EWMA a lungimilor barelor anterioare | scara barelor nu e ancorată: derivă (mers aleator în log) | `fixed_expected_ticks: H` — `E0[T] := H` constant; restul rămâne AFML |
| imbalance (TIB, VIB, DIB) | prag `E0[T]·|E[b v]|` | pe flux echilibrat pragul ≈ 0 ⇒ colaps în bare de 1 tranzacție (sau explozie) | `imbalance_threshold: rms` — pragul = RMS empiric al dezechilibrului pe ferestre de H tranzacții |

Ce rămâne neschimbat în toate variantele (principiile AFML): pragul se calculează **doar din trecut** (bare /
tranzacții anterioare), se **îngheață la deschiderea barei**, bara se închide la prima tranzacție care atinge
pragul (inclusă integral), acumularea nu se resetează la granițe de zi/fișier/batch, așteptările sunt EWMA.

---

## 1. Definițiile AFML (implementate literal)

| Tip | `v_t` | `θ_T` | Pragul înghețat `E0[θ_T]` |
|---|---|---|---|
| `tick_imbalance` (TIB) | 1 | `Σ b_t` | `E0[T]·|2P[b=1] − 1|` |
| `volume_imbalance` (VIB) | volum (bază) | `Σ b_t v_t` | `E0[T]·|2v⁺ − E0[v_t]|` |
| `notional_imbalance` (DIB) | `p_t v_t` (cotare; USDT ≠ USD) | idem | idem |
| `tick_runs` (TRB) | 1 | `max(#cumpărări, #vânzări)` în bară | `E0[T]·max(P, 1−P)` |
| `volume_runs` (VRB) | volum | `max(Σ_{b=1} v, Σ_{b=−1} v)` | `E0[T]·max(P·E0[v|b=1], (1−P)·E0[v|b=−1])` |
| `notional_runs` (DRB) | notional | idem | idem |

Bara se închide când `|θ_T| ≥ prag` (imbalance) sau `θ_T ≥ prag` (runs).

Alegeri pe care textul le lasă deschise (declarate; intră în `series_id`):

* **EWMA** cu corecție de bias (= `pandas.ewm(adjust=True)`, verificat): `num ← x + (1−α)num`,
  `den ← 1 + (1−α)den`, `α = 2/(span+1)`.
* `E0[T]`, `P[b=1]`, `E0[v|b=±1]`: EWMA peste **bare** anterioare (`ewma_span_bars`).
  `E[b v]` (imbalance): EWMA peste **tranzacții** anterioare (`ewma_span_ticks`).
* **Semnul** `b_t`: tick rule cu `b_0` = ultimul semn al barei precedente (stare purtată peste bare/zile/restart,
  resetată doar la segment nou), sau `sign: aggressor` (vezi §4).
* Tranzacțiile fără semn (începutul unui segment) numără în `T`, nu în `θ`.
* **Warm-up**: prima bară a fiecărui segment are `warmup_ticks` tranzacții și doar inițializează estimările
  (`ib_warmup = true`, `threshold_frozen` null).
* Exactitate: `θ`, sumele și `T` sunt întregi exacți; pragul e float64; depășirea int64 e refuzată (`OverflowError`).
* Gol nerecuperabil ⇒ bara deschisă invalidată, segment nou, estimări reinițializate (warm-up).

## 2. Ce nu merge în forma literală (măsurat)

### 2.1 Imbalance bars colapsează

Pe fixture-ul real, forma literală (warm-up 1000): TIB **68 132 bare, mediană 1 tranzacție** (`E0[T] → 1,0`),
VIB 50 283, DIB 50 262 (93% bare de ≤ 2 tranzacții); identic cu `sign = aggressor`. Pe date sintetice cu semne
independente, același colaps (testat).

Cauza are două părți:

1. **Valoarea de comparație e greșită pentru o regulă în valoare absolută.** Bara compară `|θ_T|` cu
   `|E[θ_T]| = E0[T]·|E[b v]|`. Pe o piață lichidă `E[b v] ≈ 0` (cumpărările și vânzările se compensează), deci
   pragul ≈ 0, în timp ce `|θ_T|` crește oricum (≈ `σ√T` din simpla fluctuație). În general `E|θ| ≥ |Eθ|`, iar
   pe flux echilibrat diferența e enormă.
2. **Bucla prin `E0[T]`.** Lungimea barei următoare ≈ `(|E[b v]|·E0[T] / σ)²`; aplicația are un punct fix
   instabil `E0[T]* ≈ σ²/E[b v]²`: sub el barele se scurtează până la 1 tranzacție, peste el se lungesc fără limită.

Stabilizatorii vechi (`expected_ticks_bounds`, `min_expected_imbalance_ratio`) mărginesc seria, dar mărimea
barelor depinde atunci în mod opac de combinația lor (pe fixture: VIB mediană 39 cu `E0[T] ∈ [200, 5000]` și
podea 0,1). Rămân disponibili, dar nu sunt recomandați.

### 2.2 Runs bars derivă

La runs bars `θ_T ≥ T/2` crește liniar, deci nu colapsează. Dar `E0[T]` este EWMA a lungimilor barelor, iar
lungimile sunt determinate de `E0[T]`: nimic nu ancorează scara. `E0[T]` face un mers aleator în scară
logaritmică, cu pas ≈ `α · CV(T)` pe bară. Pentru `span = 25` bare și un coeficient de variație de ~0,2, deviația
standard după 45 000 de bare (5 ani × 25/zi) ar fi de ordinul `e^3` — un factor ~20 în mărimea barelor.

Măsurat pe fixture (`tick_runs`, agresor, warm-up 1000, 97 bare): lungimea medie / warm-up = 0,80 în prima
jumătate, **0,62** în a doua (derivă în doar 5 ore). Cu `E0[T]` fix = 1000: 0,956 / 0,944.

## 3. Variantele recomandate (non-AFML, declarate)

### 3.1 Runs bars cu `E0[T]` fix — `fixed_expected_ticks: H`

Pragul: `H · max(P̂·Ê[v|b=1], (1−P̂)·Ê[v|b=−1])`, cu `P̂`, `Ê[v|b]` EWMA peste barele anterioare (ca în AFML).
Doar `E0[T]` e înlocuit cu constanta `H`, adică exact bucla care producea deriva. Bara se închide mai devreme
când una dintre părți e mai activă decât au arătat barele anterioare: asta e ideea AFML. Implementat ca limitarea
degenerată `[H, H]`; `ib_expected_ticks = H`. Lungimea medie rezultată pe fixture: 0,95–1,03 × H la H = 1000 și
până la 1,29 × H la H = 500 (volum); de aceea se calibrează (§5).

### 3.2 Imbalance bars cu prag RMS pe ferestre — `imbalance_threshold: rms`, `rms_horizon_ticks: H`

Definiție. Fluxul cu semn `x_t = b_t v_t` (0 pentru tranzacții fără semn) este tăiat în sub-blocuri de
`m = H/4` tranzacții, numărate de la începutul segmentului. La fiecare sub-bloc încheiat se formează o fereastră din
ultimele 4 sub-blocuri (H tranzacții; ferestrele se suprapun cu `H − m`), iar EWMA lui `(Σ_fereastră x)²` este
actualizat, cu memoria `ewma_span_ticks` exprimată în tranzacții (span în ferestre = `ewma_span_ticks / m`).
La deschiderea barei:

    prag = sqrt( EWMA[ (Σ_{fereastră de H tranzacții} b_t v_t)² ] )

adică **dezechilibrul tipic (RMS) pe H tranzacții în trecutul recent**. Bara se închide când `|θ_T| ≥ prag`.

De ce așa:

* Compară `|θ|` cu valoarea tipică a lui `|θ|` pe orizontul H, nu cu `|E[θ]|`. Pe flux echilibrat pragul ≈ `σ√H`,
  nu 0, deci colapsul dispare. Pentru semne independente, timpul mediu de ieșire din `|θ| < σ√H` este H
  (măsurat: 1,12·H la tick imbalance; 1,65·H la volum cu cozi grele, unde ultima tranzacție depășește pragul).
* Estimarea se face pe **ferestre fixe de tranzacții**, nu pe bare, deci nu depinde de unde s-au închis barele.
  Asta elimină bucla de feedback.
* Captează persistența fluxului și cozile grele ale mărimilor: dacă fluxul e în mod normal persistent, pragul e
  mai mare. Bara se închide când dezechilibrul e **neobișnuit față de trecutul recent**, nu când e doar normal
  persistent.
* Pragul rămâne cauzal și înghețat (aceleași teste de invarianță ca restul).

Măsurat pe fixture (agresor, memorie 14 000 tranzacții ≈ 1 h, warm-up 14 000): lungime medie / H = 1,40 (H = 200)
și 1,52 (H = 1000) la TIB, 2,12 / 1,61 la VIB; aceleași valori în cele două jumătăți ale eșantionului (1,48 / 1,31;
2,09 / 2,15); bare de ≤ 2 tranzacții: 0–0,8%; distribuție largă (H = 1000, TIB: p05 233, p95 5182). Variabilitatea
asta e de dorit: barele se scurtează când sosește informație.

**Varianta respinsă** (documentată pentru transparență): pragul `sqrt(μ̂²H² + σ̂²H)` cu `μ̂`, `σ̂²` estimate
per tranzacție. Pe flux persistent, zgomotul lui `μ̂` umflă termenul `μ̂²H²` proporțional cu `H²/memorie`, iar
raportul lungime/H a ieșit instabil (0,24 la H = 396 față de 0,82 la H = 1674 pe date sintetice). Calibrarea nu
convergea, așa că varianta a fost înlocuită înainte de livrare.

**Warm-up**: recomandat `warmup_ticks = ewma_span_ticks` (o memorie completă), ca primele praguri după origine
sau după un gol să nu se sprijine pe 1–2 ferestre. Dacă e mai scurt, apare un avertisment. Până la prima fereastră
completă, bara rămâne de warm-up.

Coloane: `threshold_frozen` = pragul RMS; `ib_expected_ticks` = H; `ib_expected_per_tick` = prag / H.

## 4. Semnul: agresorul observat (`sign: aggressor`)

AFML folosește tick rule pentru că, în multe date, partea agresoare lipsește. Binance raportează pentru fiecare
execuție cine a fost taker (`isBuyerMaker`), deci semnul real e disponibil. Tick rule rămâne o aproximare.
Recomandat: `aggressor`. Tick rule rămâne disponibil și produce un `series_id` diferit.

Notă de microstructură: un ordin taker care consumă mai mulți makeri apare ca mai multe rânduri `trades` cu
același timestamp și același semn. Pe fixture: 1,24 execuții per grup (timestamp, parte); autocorelația semnelor la
lag 1 este 0,41 pe execuții și 0,27 pe grupuri. Barele pe tranzacții (TIB, TRB) numără execuțiile, ca AFML.
Regula RMS absoarbe persistența mecanică în prag, iar runs bars cu `E0[T]` fix în calibrare. Agregarea pe ordine
taker nu este implementată.

## 5. Calibrare la un număr țintă de bare pe zi

Mărimea barelor e controlată de H, ca pragul la volume bars. Lungimea medie este `c·H`, unde `c` depinde de flux
(~0,95–1,3 la runs; ~1,4–2,1 la imbalance pe fixture). De aceea `scripts/calibrate_information_bars.py`:

1. ia numărul mediu de tranzacții pe zi pe toată perioada, din înregistrările de calitate ale build-urilor
   (`normalized/trades_quality`). Folosește media, nu mediana: bare/zi = tranzacții ÷ lungimea medie;
2. pentru fiecare țintă N bare/zi, vrea o lungime medie `L = tranzacții/zi ÷ N`;
3. rulează luni pilot (serii separate, sub `<root>/_calibration`, care refolosesc același cache brut), măsoară
   lungimea medie pe barele non-warm-up, rescalează `H ← H·L/medie` și repetă până la ±5% (implicit max. 3
   iterații);
4. scrie `calibration/calibration.json` (toate măsurătorile) și `calibration/target_N.args` (argumentele build-ului).

Memoria estimărilor e fixată la **~1 zi** pentru toate mărimile: `ewma_span_bars = N` (runs) și
`ewma_span_ticks = tranzacții/zi` (imbalance). Astfel cele trei dataseturi (25/50/100) compară fluxul cu același
„normal recent” și diferă doar prin orizont.

Bare/zi urmează activitatea fiecărui an, la fel ca la volume bars cu prag fix: media pe perioadă e ținta, nu
fiecare zi. `calibration.json → expected_bars_per_day_by_year` arată la ce să te aștepți pe ani.

Verificat end-to-end pe o arhivă sintetică de 92 de zile (ținte 10 și 20/zi): convergență în 2 iterații;
build-urile complete au dat 10,5–10,7 și 19,6–22,4 bare/zi.

## 6. Coloane și QA

| Coloană | Semnificație |
|---|---|
| `threshold_frozen` | pragul înghețat la deschidere, în unitatea barei (tranzacții / bază / cotare); null la warm-up |
| `ib_theta` | `θ_T` la închidere (imbalance: cu semn; runs: maximul celor două sume) |
| `ib_expected_ticks` | `E0[T]` folosit (AFML: EWMA, eventual limitat; runs fix: H; RMS: H) |
| `ib_expected_per_tick` | AFML imbalance: `|E[b v]|`; runs: `max(P·E[v⁺], (1−P)·E[v⁻])`; RMS: prag/H |
| `ib_warmup` | bară de inițializare (nu e o bară informațională) |

`manifest.json → datasets.<tip>.information_bar_qa`: bare, distribuția tranzacțiilor pe bară, ponderea barelor de
≤ 2 tranzacții, bare/zi, lungimea medie fără warm-up, `threshold_rule`, `fixed_expected_ticks` sau ponderea în care
limitele au fost active, starea curentă a estimărilor, `warning: collapsed…`.

## 7. Gărzi

* `guard_max_bars_per_day` (recomandat 5000 pentru ținte de 25–100/zi; `null` = dezactivat): o zi cu mai multe bare
  este aruncată, builderul revine la starea zilei anterioare, seria e oprită (cod de ieșire 3), celelalte continuă.
* `guard_max_open_days` (recomandat 2): avertisment dacă o bară rămâne deschisă mai mult.
* Gărzile nu intră în `series_id`.

## 8. Ce nu se garantează

* Variantele din §3 nu sunt definiția literală din carte; în lucrări, numește-le ca atare (de exemplu „runs bars
  AFML cu E0[T] fix”, „imbalance bars cu prag RMS pe ferestre de H tranzacții”).
* Comportamentul pe 5 ani a fost verificat doar pe 5 ore reale și pe date sintetice. După build-ul complet,
  verifică `information_bar_qa` și `scripts/report_datasets.py`.
* Nu se afirmă nicio proprietate statistică a barelor (normalitate, autocorelație, alpha).
* Memoria de ~1 zi, K = 4 sub-blocuri pe fereastră și ținta în medie sunt alegeri declarate, nu optime demonstrate.
