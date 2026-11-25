# Rezultatele testelor

Comandă: `.venv/bin/python -m pytest -v tests/` (Python 3.13.2, numpy 2.5.4, pyarrow 26.0.1, numba 0.68.1, polars 2.1.0).
Ultima rulare: vezi `derived/logs/pytest_full.log` (rezultat reprodus mai jos). Testele care folosesc date reale citesc
fișierele brute din 2025-06-28 05–07 (perioadă cu snapshot-uri REST la 5 minute) și stratul canonic 2025-06-28/29.

| Test | Ce demonstrează | Cerință din raport |
|---|---|---|
| `test_decimal::*` | parsare zecimală exactă în întregi scalați; valori cu mai multe zecimale semnalate, nu rotunjite | §3.1 |
| `test_bars::test_conservation_and_no_fragmentation[volume,dollar,cusum]` | fiecare tranzacție exact o dată (bare + bara deschisă), volum și notional exacte (întregi), ≤ 1 bară/tranzacție, bara nu traversează un gol, parțiale la gol | §7.1, §9.1 conservare/gap |
| `test_bars::test_volume_threshold_semantics` | închidere la prima tranzacție care atinge pragul; overshoot = cumul − prag | §7.3 |
| `test_bars::test_cusum_matches_float_recursion[3 praguri × 2 variante]` | forma exactă cu extreme = recursia `S±` din raport; resetarea doar a părții declanșate vs. ambele | §7.5 |
| `test_bars::test_cusum_equality_triggers` | egalitatea cu pragul declanșează | §7.5 pct. 5 |
| `test_bars::test_cusum_threshold_change_closes_partial_and_resets` | prag zilnic: bara deschisă se închide ca parțială și acumulatorii se resetează; apel pe zile = apel unic | §7.5 (prag adaptiv) |
| `test_bars::test_partition_independence`, `test_time_bars_chunked_equivalence` (+ bucăți de 200 k în construcție) | rezultat identic indiferent de împărțirea în bucăți/zile (stare persistentă) | §9.1 independență de partiționare |
| `test_bars::test_time_bar_boundaries` | eveniment exact la `T_{k+1}` aparține barei următoare; `eval_time = T_{k+1}−1 ns`; bara goală păstrează cursorul | §7.2, momente egale |
| `test_bars::test_real_canonical_conservation` | conservare pe o zi reală (3,6 mil. tranzacții) | §9.1 |
| `test_book::test_reconstruction_verified_by_every_snapshot` | book-ul reconstruit este egal cu fiecare snapshot REST ulterior (≥ 30 verificări, 0 diferențe) | §3.4 |
| `test_book::test_kernel_equals_reference` | kernel numba = implementare de referință pur-Python în domeniul certificat | — |
| `test_book::test_message_atomicity` | permutarea rândurilor în fiecare mesaj nu schimbă nimic | §9.1 mesaj atomic |
| `test_book::test_identical_retransmission_is_ignored` | o retransmisie identică injectată nu schimbă rezultatele | §9.1 copie identică |
| `test_book::test_conflicting_duplicate_invalidates` | un duplicat cu conținut diferit este detectat și invalidează book-ul | §3.3 |
| `test_book::test_gap_is_not_auto_recovered` | după un mesaj lipsă book-ul rămâne invalid până la un snapshot; sosirea altor mesaje nu restabilește validitatea | §3.4, §9.1 gap |
| `test_book::test_checkpoint_resume_equivalence` | reluarea din checkpoint = rulare continuă (bit cu bit) | §9.1 snapshot/checkpoint |
| `test_book::test_prefix_invariance` | rezultatele până la `t` sunt identice cu/fără date ulterioare | §9.1 prefix |
| `test_book::test_identities` | `wm−m = s/2·QI1`, imbalance ∈ [−1,1], cost monoton în mărime, spread > 0 | §9.1 identități |
| `test_book::test_tw_spread_and_ofi_exact` | medii temporale și OFI exacte pe un flux sintetic calculat manual; fereastra `(t−w, t]` exclude evenimentul de la `t−w` | §5.1, F5 |
| `test_book::test_skip_unanchored_is_exact` | omiterea decodării orelor fără snapshot când book-ul este neancorat nu schimbă rezultatele | optimizare |
| `test_book::test_far_levels_outside_tick_grid_are_ignored_safely` | niveluri de checkpoint la prețuri în afara grilei (până la 4,5 mil. USDT) nu produc scrieri în afara limitelor și nu extind domeniul | incident 2026-09-21 |
| `test_book::test_outputs_independent_of_query_set` | valorile la un moment dat sunt identice bit cu bit indiferent ce alte interogări (alte seturi de bare) se evaluează în același replay | comparabilitate între configurații |
| `test_features::test_trade_features_bruteforce` | volum, TFI, VWAP, range, număr, cuantile, randamente = calcul direct | F1, F3 |
| `test_features::test_rv_identities` | RV = RV⁺ + RV⁻, RV ≥ 0 | §9.1 identități |
| `test_features::test_trade_prefix_invariance` | features din tranzacții la `t` identice când datele se opresc la `t` | §9.1 prefix |
| `test_features::test_oi_no_lookahead` | OI as-of folosește doar versiuni recepționate ≤ `t` | F8 |
| `test_features::test_tw_piecewise_exact` | medie temporală exactă cu validitate expirată în fereastră | §5.1 |
| `test_features::test_markout_maturation` | markout-ul apare numai după maturizare; effective spread folosește cotația strict anterioară | F6, §9.1 maturizare |
| `test_registry::*` | fiecare coloană emisă are exact o definiție în registru | §5.2 |

| `test_book::test_snapshot_written_in_two_halves_is_one_message`, `test_identical_snapshot_copies_keep_earliest_real_reception` | părți de snapshot cu nivele diferite = un mesaj, disponibil la ultima parte; copiile identice păstrează prima recepție reală | particularități furnizor |
| `test_book::test_snapshot_connects_to_messages_received_before_it`, `test_snapshot_bridges_unpublished_update` | Defectele A și B; aplicarea nu are loc înainte de disponibilitatea snapshot-ului | D1 |
| `test_book::test_rebuilt_top_certificate_only_when_no_hidden_tick_remains`, `test_rebuilt_certificate_uses_reported_ticks_between_best_levels` | certificatul fără ancoră | D1 |
| `test_book::test_rebuilt_restarts_empty_after_a_true_gap`, `test_rebuilt_burst_of_holes_costs_one_timeout` | după un gol real nu supraviețuiește nimic; o cascadă costă o așteptare | D1 |
| `test_book::test_rebuilt_book_returns_to_anchored_tier_at_a_snapshot`, `test_empty_book_independent_of_snapshots_in_probe_mode`, `test_strict_mode_never_starts_from_an_empty_book` | separarea nivelurilor; modul strict nu pornește din gol | D1 |
| `test_book::test_rebuilt_certified_values_equal_the_anchored_book` | pe date reale, tot ce emite book-ul pornit gol e egal cu book-ul ancorat | D1 |
| `test_book::test_rebuilt_crossing_opens_a_quarantine` | după un flux încrucișat nu se emite nimic | D1 |

| `test_features::test_oi_*`, `test_rolling_statistics_ignore_points_not_yet_received` | clasificarea versiunilor OI, carantina fără alegere, propagarea, disponibilitatea la recepție | D2 |
| `test_features::test_hour_without_liquidation_file_is_unknown_never_zero`, `test_liquidation_unknown_propagates_to_combinations`, `test_shared_gap_makes_time_since_unknown` | ore fără fișier: necunoscut, niciodată zero | D3 |
| `test_features::test_funding_*` | vechimea capătului, pană fără valori inventate, interval necunoscut după o tăcere lungă | mark/funding |
| `test_features::test_return_ignores_gaps_inside_the_window_but_not_at_its_ends` | randamentul depinde doar de două prețuri, fluxul rămâne strict | D4 |
| `test_book::test_markouts_do_not_cross_a_batch_boundary_early`, `test_rebuilt_checkpoint_resume_equivalence`, `test_rebuilt_crossing_voids_the_certificate_until_it_is_earned_again` | granițe de fișier, reluare, certificare după încrucișare | D5, D1 |

## Rezultat

```
77 passed  (vezi derived/logs/pytest_regen.log)
```

Regresie strict: `scripts/d1_strict_regression.py`, 9.013.898 valori din 2025-06-28/29 identice bit cu bit cu datasetul existent.

## Verificări pe istoricul complet (nu teste unitare, ci execuții)

- Conservare pe 1.771.277.674 tranzacții economice: pentru fiecare set de bare cu prag fix, suma tranzacțiilor,
  volumului (10⁻³ BTC) și notionalului (10⁻⁵ USDT) din bare + bara deschisă = fluxul canonic, exact
  (`derived/audit/bars_summary.csv`). Seturile adaptive: diferența = exact cele 12.763.179 tranzacții din cele 7 zile
  de warm-up (fără prag).
- Reconstrucția L2 contra snapshot-urilor pe întreg replay-ul: vezi `docs/VALIDITY_MAP.md` / `book_state/…/counters_*.json`.

- Verificarea independentă a datasetului asamblat (`scripts/verify_dataset.py`, cod separat de calculul features):
  5 zile reprezentative × 9 seturi (2025-06-29 book valid, 2025-12-15 neancorat, 2026-04-24 întreruperea mark/ticker,
  2026-10-02 checkpoint-uri orare, 2026-10-16 după anomalia OI) — 97.456 bare, 16.278 recalculări brute-force ale
  volumului/TFI/VWAP: 0 erori; 0 nepotriviri de conservare; 0 valori prezente sub măști invalide; 0 încălcări de
  ordine temporală sau de interval; 0 coloane neînregistrate (`derived/datasets/btcusdt_initial/verification.json`).
  Demo 2025-06-28 → 30 (6 seturi): aceleași verificări, 0 erori (`derived/datasets/demo_3days/verification.json`).
- Reproductibilitate: demo-ul regenerat separat coincide cu datasetul complet pe zilele comune (după corecția
  integrării independente de setul de interogări; înainte, 90 de coloane temporal-ponderate difereau cu ≤ 3e−6
  relativ din cauza ordinii de însumare float — corectat, testat bit cu bit, replay refăcut).
