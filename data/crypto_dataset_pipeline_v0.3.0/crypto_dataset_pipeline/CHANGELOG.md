# Changelog

## 0.3.0 — 2026-10-26

### Information-driven bars: variante corecte pentru dataseturi lungi (declarate, non-AFML)

* **Imbalance bars cu prag RMS pe ferestre** (`imbalance_threshold: rms`, `rms_horizon_ticks: H`): pragul este
  dezechilibrul tipic (RMS) al fluxului cu semn pe ferestre de H tranzacții din trecutul recent, estimat pe
  ferestre fixe de tranzacții, nu pe bare. Elimină colapsul formei literale. Rămâne cauzal și înghețat la
  deschidere. Pe ETHUSDT real: 0–0,8% bare de ≤ 2 tranzacții, raport lungime/H stabil.
* **Runs bars cu `E0[T]` fix** (`fixed_expected_ticks: H`): elimină deriva scării barelor. Pe fixture, forma literală
  a derivat de la 0,80 la 0,62 × warm-up în 5 ore; varianta fixă a rămas la 0,96 / 0,94.
* **Parametri per tip** (`bars.information.per_type`): un singur build produce toate cele 6 tipuri, fiecare cu
  parametrii lui.
* **Identitatea seriei** conține exact parametrii care schimbă barele fiecărui tip. Identificatorii time / volume /
  notional și AFML imbalance din 0.2.0 sunt neschimbați. Cei ai runs bars s-au schimbat: nu mai depind de
  `ewma_span_ticks` și de podea, care nu le afectau.
* Varianta i.i.d. `sqrt(μ²H² + σ²H)`, testată în timpul dezvoltării, a fost respinsă: calibrarea nu convergea pe
  flux persistent (`docs/INFORMATION_BARS.md` §3.2).
* QA: `mean_ticks_per_bar_excl_warmup`, `threshold_rule`, `fixed_expected_ticks` / `bound_binding_share`.
  Avertisment dacă warm-up-ul regulii RMS e mai scurt decât memoria ei.
* Stările salvate de 0.2.0 se încarcă în continuare (sloturi noi completate); continuarea e identică (testat).

### Scripturi

* `scripts/eth_datasets.sh`: rularea completă ETHUSDT (download → volume/notional → calibrare → bare de informație
  → raport), pe pași reluabili, configurabil prin variabile de mediu.
* `scripts/calibrate_information_bars.py`: calibrează H la bare/zi țintă pe luni pilot (serii separate) și scrie
  `calibration.json` și `target_N.args`.
* `scripts/report_datasets.py`: tabelul tuturor dataseturilor (`reports/datasets_summary.csv`).
* `docs/RUN_ETH.md`: ghidul pas cu pas.
* `configs/eth_spot.yaml`: semnul = agresorul observat, regula RMS, gărzi pentru ținte de 25–100 bare/zi.
* Eșantionul real conține acum cele 9 tipuri de bare, cu variantele recomandate.

## 0.2.0 — 2026-10-22

### Nou: information-driven bars (AFML 2018, §2.3.2)

* Tipuri noi în `bars.types`: `tick_imbalance`, `volume_imbalance`, `notional_imbalance`, `tick_runs`,
  `volume_runs`, `notional_runs`, în același kernel exact și restartabil ca volume/notional bars (fără reset la
  granițe de zi/fișier/batch; stare persistată, inclusiv EWMA-urile, în hex exact).
* Prag `E0[θ_T]` înghețat la deschiderea barei, din EWMA-uri (adjust=True) ale barelor anterioare; tick rule cu
  `b_0` purtat din bara precedentă; bară de warm-up la începutul fiecărui segment; variantă separată
  `sign: aggressor`.
* Coloane noi (numai pentru aceste tipuri): `ib_theta`, `ib_expected_ticks`, `ib_expected_per_tick`, `ib_warmup`;
  `threshold_frozen` = `E0[θ_T]`. QA de formă în manifest (`information_bar_qa`).
* Stabilizatoare opționale, declarate ca **non-AFML**: `expected_ticks_bounds`, `min_expected_imbalance_ratio`.
* Gardă de colaps (`guard_max_bars_per_day`): ziua respectivă e aruncată, seria oprită, celelalte continuă;
  avertisment pentru bare care nu se mai închid (`guard_max_open_days`).
* Constatare pe date reale ETHUSDT: runs bars stabile (scara lor urmează `warmup_ticks`); imbalance bars literale
  colapsează în bare de o tranzacție. Detalii și cifre: `docs/INFORMATION_BARS.md`.
* Eșantionul real (`sample_output/`) conține acum și `tick_runs`, `volume_runs`, `notional_runs`.

### Validări (din diagnosticul build-urilor ETHUSDT)

* `instrument.symbol` trebuie să fie egal cu `base + quote` (cauza datasetelor ETH construite cu base = BTC).
  Nou: `configs/eth_spot.yaml`.
* `features.dvol.max_age` trebuie să fie ≥ rezoluția DVOL.
* **Zile lipsă:** o zi fără 404 confirmat (descărcare eșuată, cache incomplet, rețea) nu mai devine un gol
  permanent în starea barelor: construcția se oprește înaintea ei (`bars.on_unconfirmed_missing_day: stop`), iar
  după ce ziua e obținută rezultatul este identic cu un build fără problemă. Un 404 pentru o zi mai veche decât
  `archive_publication_lag_days` este memorat (`.notfound.json`) și produce gol + segment nou, ca înainte.
  Zilele din lag-ul de publicare dau `status = complete_until_publication_lag` (cod 0).
* Detectarea golurilor „învechite”: un gol din starea salvată care acoperă zile prezente acum în cache produce un
  avertisment cu instrucțiunea de reconstrucție.
* Avertismente: `bar_origin == start`, warm-up intraday mai scurt decât cea mai lungă fereastră, imbalance bars
  literale, skew pe alt activ decât cel de bază (în stderr și în `manifest.json → warnings`).
* Manifest/rezumat: `status`, `stopped_before_day`, `halted_series`, `warnings`; `missing_trade_days` are acum
  forma `[{day, reason}]`.
* CLI: cod de ieșire `2` pentru erori de configurație și `3` pentru build incomplet.
* Sumele int64 pe bară sunt verificate: o depășire (ex. time bars lungi pe BTC spot la scara 10⁻¹⁰ USDT) ridică
  `OverflowError` în loc să producă o valoare înfășurată.

### Compatibilitate

* `series_id` pentru time / volume / notional este neschimbat (aceleași granițe, verificat pe eșantion).
* Build-urile vechi construite peste zile lipsă neconfirmate trebuie reconstruite: ștergeți `data/state/` și
  `data/normalized/px1s/`, rulați `download`, apoi `build`.
