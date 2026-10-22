# Registrul formulelor (AFML și convenții comune)

## 0. Starea verificării față de *Advances in Financial Machine Learning* (Wiley, 2018)

* **Ce s-a putut verifica (2026-10-20):** pagina O'Reilly a capitolului 17 (link-ul din cerință) afișează doar
  previzualizarea secțiunilor 17.1–17.2 și cuprinsul (titlurile „17.3 CUSUM Tests” și „17.4 Explosiveness Tests”).
  **Nicio formulă din carte nu este vizibilă** în previzualizare. Capitolele 2 și 19 nu au putut fi consultate.
* **Ce NU s-a făcut:** nu s-au folosit copii neautorizate ale cărții; nu sunt citate numere de pagină sau snippet-uri.
* **Consecință:** formulele implementate sunt exact cele scrise în specificația sarcinii (secțiunile 5.1–5.8), cu
  convențiile lipsă alese explicit și documentate mai jos. Ele au fost verificate numeric față de implementări de
  referință independente (OLS `statsmodels`, căutare exhaustivă brute-force), nu față de textul cărții.
  Implementările terțe (MizarLabs) au fost citite doar pentru comparație.
* **Comutator:** `features.structural.formula_gate`:
  * `task_spec` (implicit) — se emit statisticile definite de specificația sarcinii, sub nume care descriu
    configurația (fereastră, τ, laguri), fără a pretinde fidelitate verificată față de carte;
  * `primary_verified_only` — toate coloanele structurale devin null cu motivul `unverified_formula`
    (pentru utilizatori care cer fidelitate certificată față de textul primar).

Acest statut apare și în câmpul `reference` al fiecărei coloane din registrul JSON.

## 1. Seria de intrare comună pentru statisticile structurale

* `y_t = log(P_t)`, unde `P_t` este close-ul *previous-tick* pe clock-ul barelor: close-ul barei dacă are
  tranzacții; pentru o bară de timp completă fără tranzacții, ultimul preț executat anterior din același segment
  (feedul complet arată că nu s-a tranzacționat). O bară incompletă (gol) sau o bară invalidată rupe lanțul;
  ferestrele nu traversează niciodată o ruptură sau un segment nou.
* Clock: bara de evenimente sau bara de timp, după dataset. Fereastra de W bare acoperă durate economice diferite pe
  bare de volum/notional; durata barelor este livrată separat (`bar_duration_s`). Pragurile critice calibrate pentru
  alte procese de eșantionare **nu** sunt transferate (nu se emit).
* Disponibilitate: statistica pentru endpointul t folosește numai observații ≤ t și devine disponibilă la
  `decision_time` al barei t. Nimic nu este atribuit retroactiv unei date anterioare.

## 2. CUSUM Brown–Durbin–Evans — `bde_cusum_w{W}_p{p}`

| Câmp | Valoare |
|---|---|
| Referință | AFML 2018, cap. 17 („CUSUM Tests”, titlu verificat); formă = cerința §5.1 |
| Model (configurația testului) | `r_i = c + Σ_{j=1..p} φ_j r_{i−j} + ε_i`, `r_i = y_i − y_{i−1}` (randament log pe bară); implicit p = 1 |
| Reziduuri recursive | pentru i după primele k = 1+p observații ale ferestrei: `w_i = (r_i − x_iᵀ β̂_{i−1}) / sqrt(1 + x_iᵀ (X_{i−1}ᵀX_{i−1})⁻¹ x_i)`, cu β̂_{i−1} estimat **numai** pe observațiile anterioare lui i din fereastră |
| Standardizare | `S_t = Σ_j w_j / s_w`, `s_w` = abaterea standard eșantională (ddof = 1, cu centrare) a reziduurilor disponibile la t — niciodată pe eșantionul viitor |
| Fereastră | ultimele W bare ale lanțului curent (implicit 200); variantă *windowed* |
| Eroare standard | — (statistică cumulativă); sub H0 `S_t ~ N(0, n_w)`, `n_w` livrat ca `bde_n_resid_*` |
| Benzi | nu se emit (depind de lungimea totală a eșantionului T, care la evaluare cauzală este t) |
| Diferențe față de biblioteci | `statsmodels.recursive_olsresiduals`: aceeași `w` (`rresid_scaled`) și aceeași normalizare (`var(ddof=1)` pe `rresid_scaled[skip:]`); folosită însă pe întreaga serie, deci necauzal dacă se reutilizează ca feature. Testul compară valoarea finală a ferestrei cu `rcusum[-1]`: egalitate la 1e-9. Normalizarea exactă din AFML (inclusiv eventualul σ̂_ε în f_t) nu a putut fi verificată. |
| Nu se confundă cu | filtrul CUSUM simetric pentru selecția evenimentelor (nu este implementat aici) |

## 3. CUSUM Chu–Stinchcombe–White — `csw_up_w{W}`, `csw_down_w{W}`, `csw_two_w{W}`

| Câmp | Valoare |
|---|---|
| Referință | AFML 2018, cap. 17 („CUSUM Tests”); formă = cerința §5.2 |
| Serie | `y` = log-preț (configurație declarată) |
| Statistică | `S_{n,t} = (y_t − y_n) / (σ̂_t sqrt(t − n))`, `σ̂_t² = (t − 1)⁻¹ Σ_{i=2..t} (Δy_i)²`; indicii sunt relativi la începutul ferestrei (eșantionul disponibil ales, i = 1..W) |
| Agregare | supremum peste n ∈ [1, t−1] cu endpoint fix t; separat: creștere `sup S`, scădere `sup (−S)`, bilateral `sup |S|` |
| Fereastră | W bare (implicit 200), numită în coloană |
| Frontiere critice | nu se emit: constanta b_α nu a fost verificată pentru acest proces de eșantionare |
| Diferențe | articolul MQL5 (implementare terță) folosește aceleași formule; nu certifică fidelitatea |

## 4. SADF (windowed, start-uri înapoi la endpoint fix) — `sadf_w{W}_tau{τ}_p{p}_c`

| Câmp | Valoare |
|---|---|
| Referință | AFML 2018, cap. 17 („Explosiveness Tests”); formă = cerința §5.3 |
| Regresie | `Δy_i = α + β y_{i−1} + Σ_{j=1..p} φ_j Δy_{i−j} (+ γ i dacă adf_trend) + ε_i` pentru rândurile i ∈ [t0+p+1, t] |
| Statistică | `ADF_{t0,t} = β̂ / SE(β̂)`, `SE² = s² [(XᵀX)⁻¹]_{ββ}`, `s² = RSS / (n_eff − k)` (OLS clasic) |
| Grilă de începuturi | **toate** t0 cu `t0 ≥ max(început lanț, t − W + 1)` și `n_eff = t − t0 − p ≥ τ` (implicit τ = 30); fără subeșantionare |
| Agregare | `SADF_t = max_{t0} ADF_{t0,t}` (coada dreaptă; nu valoarea absolută) |
| Nomenclatură | statistica la endpoint fix cu supremum peste începuturi = ce PSY numesc **BSADF_t** și ce AFML numește SADF_t; limitarea la W bare ⇒ variantă *windowed/rolling*, marcată prin `_w{W}` |
| Laguri / trend | fixate prin configurație (implicit p = 1, intercept, fără trend); nu se selectează pe dataset |
| Numeric | rândurile sunt adăugate incremental pe măsură ce t0 se mută înapoi (XᵀX, Xᵀy), cu centrare pe y_t (invariant pentru modelul cu intercept); rezultatul este egal cu OLS brute-force (test, toleranță relativă 1e-8) |
| Diferențe față de MizarLabs | MizarLabs iterează start-urile 0…len−min_length pe toată seria (fereastră expansivă), indexează rezultatul la `y.index[min_length:]` și nu are limită W |

## 5. QADF și CADF — `qadf_q{q}_…`, `cadf_q{q}_…`

| Câmp | Valoare |
|---|---|
| Referință | AFML 2018, cap. 17; formă = cerința §5.5 |
| Mulțime | aceleași statistici `{ADF_{t0,t}}` ca la SADF (subferestre care se termină la t) |
| QADF | cuantila q (implicit 0.95) cu interpolare liniară între statisticile de ordine (Hyndman–Fan tip 7, implicitul numpy) |
| CADF | media statisticilor `ADF ≥ QADF_t` (coadă **inclusivă**; egalitățile la cuantilă sunt incluse); coada nu e niciodată goală, deoarece maximul ≥ cuantila |
| Notă | convenția exactă AFML (≥ vs >, tipul cuantilei) nu a putut fi verificată; q nu se optimizează pe date |

## 6. SDFC (Chow-type Dickey–Fuller) — `sdfc_w{W}_trim{τ0}`, `sdfc_argmax_bars_ago_w{W}`

| Câmp | Valoare |
|---|---|
| Referință | AFML 2018, cap. 17; formă = cerința §5.4 |
| Regresie | `Δy_i = δ y_{i−1} D_i(τ) + ε_i`, `D_i(τ) = 1{i ≥ τ}`, **fără intercept, fără laguri**, pe fereastra ultimelor W puncte (indici relativi 1..W; observații i = 2..W) |
| Candidați | τ ∈ [⌈τ0·W⌉, ⌊(1−τ0)·W⌋] (și τ ≥ 2), implicit τ0 = 0.15 |
| Statistică | `DFC_τ = δ̂ / SE(δ̂)`, `s² = RSS/(N − 1)`, N = W − 1; `SDFC_t = max_τ DFC_τ` |
| Indicator | 1{i ≥ τ} (convenția din MizarLabs și din sursele secundare); 1{i > τ} din cerință este aceeași familie decalată cu un indice |
| Capcana retrospectivă | valoarea este stocată **doar la endpointul t**; `sdfc_argmax_bars_ago` dă distanța (în bare) până la candidatul maxim, cunoscută la t. Testul verifică că trunchierea seriei la un candidat τ reproduce exact rândul τ |
| Diferențe față de MizarLabs | MizarLabs returnează vectorul DFC indexat la data candidatului (cu observații ulterioare acelei date) și lasă supremumul utilizatorului; candidații sunt limitați printr-un număr absolut de observații |

## 7. Kyle λ — `kyle_lambda_trades_{tick|aggr}`, `kyle_lambda_bars{W}_{aggr|tick}`

| Câmp | Valoare |
|---|---|
| Referință | AFML 2018, cap. 19 (neconsultat); formă = cerința §5.6 |
| Nivel tranzacție (în bară) | `Δp_i = λ b_i q_i + ε_i`, fără intercept; `Δp_i` în moneda de cotare, `q_i` în unități de bază; tranzacția i = toate execuțiile barei, Δp-ul primei tranzacții folosește tranzacția anterioară (din bara precedentă, dacă există în același segment) |
| Semn | `tick` = tick rule (varianta literală); `aggr` = agresor observat (variantă separată, nume diferit) |
| Bare (rolling W) | `P_B − P_{B−1} = λ Σ_{i∈B} b_i q_i + ε_B`; regresorul este suma volumelor semnate ale tranzacțiilor (nu `sign(close−open)·volume`) |
| Eroare standard | clasică: `s² = RSS/(n−1)`, `SE = sqrt(s²/Σx²)`, `t = λ/SE` (coeficient / rădăcina varianței; testat) |
| Null | n < `min_trades_regression`, Σx² = 0, RSS ≤ 0; λ negativ nu este tăiat |
| Unități | USDT per BTC de volum semnat |

## 8. Hasbrouck λ — `hasbrouck_lambda_trades_{tick|aggr}`, `hasbrouck_lambda_bars{W}_{aggr|tick}`

| Câmp | Valoare |
|---|---|
| Formă | `r_i = λ b_i sqrt(p_i q_i) + ε_i`, `r_i = log(p_i/p_{i−1})`, fără intercept (cerința §5.7) |
| Bare | `r_B = λ X_B`, `X_B = Σ_{i∈B} b_i sqrt(p_i q_i)` (suma rădăcinilor per tranzacție — testat că diferă de `sign·sqrt(Σ p q)`) |
| Unități | randament log per sqrt(USDT) |
| Note | cu tranzacții agregate (aggTrades) rădăcina se aplică mesajului agregat, nu execuțiilor — de aceea sursa implicită este `trades` |

## 9. Tick rule și imbalance din agresor

* `b_i = +1` dacă `p_i > p_{i−1}`, `−1` dacă `p_i < p_{i−1}`, altfel `b_{i−1}`. Starea (ultimul preț și ultimul semn)
  este purtată între fișiere, batch-uri și restart; se resetează doar la un segment nou după un gol. Tranzacțiile de
  dinaintea primei schimbări de preț rămân cu semn necunoscut (nu se completează retroactiv).
* Agresor Binance: `isBuyerMaker = true` ⇒ vânzătorul a fost agresor ⇒ −1; `false` ⇒ +1.
* `I_B = (V_buy − V_sell)/(V_buy + V_sell)` în volum de bază și în notional; null dacă există volum neclasificat
  (`aggr_classified_frac_*` expune acoperirea) sau numitor zero.

## 10. Convenții comune

* **z-score cauzal:** `z_t = (x_t − mean(x_{t−N..t−1})) / sd(x_{t−N..t−1})`, ddof = 1, N observații **valide**
  anterioare (observațiile lipsă sunt sărite, niciodată completate), minim configurat; sd = 0 ⇒ null.
  Premium zilnic: N = 90 de zile valide (min 60). ETF: N = 60 de sesiuni cu valori (min 40). Valoarea curentă este exclusă.
* **As-of:** numai înapoi, pe `available_at`; observațiile cu `available_at == decision_time` din alte fluxuri sunt
  considerate necunoscute (`cross_stream_tie: exclude_equal`). Vechimea este livrată (`*_age_s`) și limitată (`max_age`).
* **Log-uri:** intrări nepozitive ⇒ null (`nonpositive_log_input`); fără epsilon, clipping sau imputare.
* **Volatilitate realizată:** `RV_W = Σ r_g²` pe grila de 1 min (preț previous-tick strict înaintea punctului de
  grilă, vechime ≤ 5 min); `vol = sqrt(RV)`; `vol_ann = sqrt(RV · T_year/T_W)` cu T_year = 365 zile (piață 24/7,
  convenție declarată). Pe bare de evenimente nu se anualizează.
* **Trend 30 zile:** `log(C_d/C_{d−30}) / (s_30 sqrt(30))`, `s_30` = sd(ddof=1) a celor 30 de randamente log zilnice
  încheiate la d; necesită 31 de închideri UTC consecutive.
* **Skew 25Δ, 30 zile:** interpolare liniară a IV în delta între cele două cotații care încadrează ±0.25 (fără
  extrapolare), apoi interpolare liniară a varianței totale `w = σ²τ` între scadențele care încadrează 30 de zile;
  `skew = IV_put(−25Δ) − IV_call(+25Δ)`.
* **Cost din carnet:** cumpărare = consumă ask-urile, vânzare = bid-urile, ultimul nivel parțial;
  `slippage_buy = 1e4 (VWAP/m − 1)`, `slippage_sell = 1e4 (1 − VWAP/m)` față de mid (include traversarea spread-ului
  o singură dată); adâncime insuficientă în zona cunoscută complet ⇒ null; comision numai dacă e configurat.
