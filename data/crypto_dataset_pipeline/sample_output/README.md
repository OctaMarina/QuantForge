# Eșantion real mic

Construit cu `python crypto_dataset.py sample-real --config configs/sample_real_ethusdt.yaml` din tranzacțiile reale
Binance ETHUSDT spot din 2020-08-14, 10:00–15:00 UTC (fixture public, vezi `tests/fixtures/real/NOTICE.md`).

* `data/datasets/<config_hash>/` — ieșirea completă a pipeline-ului: `time.parquet` (59 bare de 5 min închise; prima
  e marcată incompletă deoarece nu există dovada completitudinii înainte de prima tranzacție), `volume.parquet`
  (94 bare de 2000 ETH), `notional.parquet` (80 bare de 1 000 000 USDT), barele parțiale, calitatea (motive de
  null), acoperirea pe feature × lună, registrele coloanelor, `manifest.json`.
* `*_preview.csv` — o selecție de coloane pentru inspecție rapidă.
* `coverage_summary.csv` — valori valide / rânduri și motivul principal de null, pe feature.
* `build_summary.json` — ieșirea comenzii.

Pragurile și ferestrele sunt micșorate pentru 5 ore de date (ilustrative, neoptimizate). Familiile fără sursă în
acest mediu (premium, basis, funding, ETF, Fear & Greed, DVOL, skew, lichidări, L2) sunt prezente ca **coloane null
cu motiv explicit**, nu omise și nu completate.
