# Runbook — generare, reluare, extindere, configurații noi

Toate comenzile se rulează din `~/Desktop/QuantForge/data/cryptohftdata` cu `.venv/bin/chdds` (sau
`.venv/bin/python -m chdds.cli`). Nimic nu scrie în `cryptohftdata_raw/`.

## 1. Pipeline complet (istoric)

```bash
C=configs/btcusdt_initial.yaml
.venv/bin/chdds audit --workers 9        # ~75 min pentru .zst + ~35 min pentru .parquet (22 nuclee, 30 GB RAM)
.venv/bin/chdds consolidate
.venv/bin/chdds canonical
.venv/bin/chdds bars --config $C
.venv/bin/chdds features --config $C --workers 8
.venv/bin/chdds book --config $C --workers 6
.venv/bin/chdds assemble --config $C
.venv/bin/chdds audit-report
```

Durate și memorie măsurate: `docs/BENCHMARK.md`.

## 2. Reluare sigură

| Etapă | Mecanism | Ce faci după o întrerupere |
|---|---|---|
| audit | fișierul JSON per fișier brut este scris atomic la final; se sare peste fișierele cu aceeași (dimensiune, mtime) | rulezi aceeași comandă |
| canonical | idempotent (rescrie zilele) | rulezi aceeași comandă |
| bars | `derived/bars/<spec>/_state/<zi>.npz` după fiecare zi; continuă după ultima zi construită | rulezi aceeași comandă |
| features | un fișier per zi și set de bare, scris atomic; zilele existente sunt sărite | rulezi aceeași comandă |
| book | checkpoint zilnic `derived/book_state/<cfg>/<zi>/` (stare L2, inele, resolver, buffer) | `chdds book --config C --resume-from 2026-05-31 --set features.book_start=2026-06-01` |
| assemble | rescrie lunile | rulezi aceeași comandă |

Echivalența reluării cu rularea continuă este testată (`tests/test_book.py::test_checkpoint_resume_equivalence`,
`tests/test_bars.py::test_partition_independence`).

## 3. Extindere cu date noi (de ex. noiembrie 2026)

1. Descarci fișierele noi în `cryptohftdata_raw/` (downloader-ul existent).
2. `chdds audit` (procesează doar fișierele noi), `chdds consolidate`.
3. `chdds canonical --start 2025-06-28 --end 2026-11-30` (seq global se recalculează determinist).
4. `chdds bars --config C --set dataset.end=2026-11-30` — continuă din starea zilei 2026-10-31; rândurile publicate
   nu se rescriu. Bara deschisă din `_partial_end_<end>.parquet` este regenerată.
5. `chdds features ...`, `chdds book --resume-from 2026-10-31 --set features.book_start=2026-11-01`, `chdds assemble`.

## 4. Configurații noi (alte praguri/ferestre) fără rescrierea pipeline-ului

Un set de bare este identificat prin `spec_id = <nume>-<hash(spec normalizat + versiune convenții)>`; schimbarea
oricărui parametru semantic produce alt `spec_id` și alt director, deci variantele se pot compara.

```yaml
bars:
  - {name: volume_100btc, kind: volume, threshold: 100}
  - {name: cusum_2bps_both, kind: cusum, threshold: 2, cusum_reset: both}
  - {name: dollar_adapt5m, kind: dollar, adaptive: {target_s: 300, lookback_days: 14}}
features:
  families: [bar, trades, mark, oi, liq, ticker, calendar, book, combos]
  trades: {flow_windows_s: [10, 60, 600], quantile_windows_s: [60]}
  slow: {liq_missing_file_policy: unknown}
  book: {bands_bps: [2, 10, 25], windows_s: [1, 10, 60], max_age_ms: 500}
  select: ["bar_*", "tfi_*", "ofi_*", "basis_*"]     # selecția coloanelor pentru experiment
```

Bare și features din tranzacții/fluxuri lente: minute–zeci de minute pe istoric. Features de book: necesită un
replay L2 (ore) — un replay servește simultan toate seturile de bare din aceeași configurație. Grila de 100 ms
(`book_grid`) rămâne disponibilă pentru features pe mid fără un nou replay.

## 5. Teste

```bash
.venv/bin/python -m pytest -q            # ~3-4 min; folosește datele reale din 2025-06-28/29 dacă există
```

## 6. Experimente de diagnostic (nu intră în dataseturile implicite)

`scripts/heal_experiment.py` — măsoară cât de greșit este un book „vindecat” după goluri, folosind snapshot-urile
ca sonde. Înlocuit de experimentul D1.

Experimentul D1 și ordinea pașilor: `docs/D1_REPORT.md` §13. Politica se fixează pe calibrare înainte de a rula tabelele de validare.

## 7. Modul opțional reconstruit

Se activează din configurație și scrie sub alt nume de dataset, fără să atingă dataseturile strict:

```yaml
features:
  book:
    book_mode: rebuilt
    checkpoint_cert_levels: 500
```

```bash
bash scripts/run_rebuilt_demo.sh      # demonstrația pe 14–16.12.2025 și 9–11.10.2025
```

Pentru reproducerea exactă a dataseturilor strict existente: `--set features.book.legacy_chain=true`. Checkpoint-urile vechi de
replay se pot relua (câmpurile noi pornesc de la zero).

## 8. Regenerare

`docs/REGEN_READINESS.md` descrie starea. Rularea: `systemd-run --user --unit=regen --property=MemoryMax=20G bash scripts/run_regen.sh`. Proba pe ferestre
selectate: `bash scripts/run_regen_dry.sh`, apoi `scripts/verify_regen_dry.py`. Dacă rezolverul strict vechi e necesar pentru reproducere: `--set features.book.legacy_chain=true`.

