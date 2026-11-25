# Dataset `btcusdt_v2`: rezultate măsurate

Generat de `scripts/dataset_summary.py`. Locație: `derived/datasets/btcusdt_v2/`, 34G, 9.413.305 bare, 9 seturi. Mod `rebuilt`, rezolver corectat. Datasetul `btcusdt_initial` și datele brute sunt neatinse.

Procentele sunt cote din barele setului cu valoare prezentă, media pe coloanele familiei. Identitatea barei, coloanele `*_state` și metadatele de calitate nu intră în medie.

## Bare

| Set | Bare | Coloane | Coloane pentru model |
|---|---|---|---|
| time_1m | 707.040 | 664 | 643 |
| time_5m | 141.408 | 664 | 643 |
| volume_50btc | 1.435.551 | 666 | 645 |
| volume_adapt60s | 1.390.444 | 666 | 645 |
| dollar_5musdt | 1.220.171 | 666 | 645 |
| dollar_adapt60s | 1.386.082 | 666 | 645 |
| cusum_5bps | 1.503.182 | 670 | 649 |
| cusum_10bps | 462.400 | 670 | 649 |
| cusum_adapt_c1 | 1.167.027 | 670 | 649 |

## Verificări

- Book: 3.314 verificări cu snapshot-uri, 0 eșuate, 0 niveluri nepotrivite; 7.166 ancore, 1.929 porniri din gol, 29 încrucișări (toate pe 2025-08-29).
- `verify_dataset.py` pe 9 zile: 251.928 bare, 42.445 recalculări brute-force, 0 erori, 0 erori de conservare, 0 valori sub măști invalide, 0 coloane neînregistrate.

## Proveniența book-ului (cotă din bare)

**Tot istoricul**

| Stare | time_1m | time_5m | volume_50btc | volume_adapt60s | dollar_5musdt | dollar_adapt60s | cusum_5bps | cusum_10bps | cusum_adapt_c1 |
|---|---|---|---|---|---|---|---|---|---|
| altele | 0,0 | 0,0 | 0,0 | 0,0 | 0,0 | 0,0 | 0,1 | 0,1 | 0,1 |
| ancorat (0) | 40,6 | 40,6 | 38,4 | 40,7 | 33,0 | 41,0 | 32,4 | 31,3 | 38,2 |
| fără book (1) | 1,8 | 1,8 | 1,7 | 1,6 | 1,4 | 1,6 | 1,5 | 1,4 | 1,5 |
| necertificat (10) | 0,0 | – | 0,0 | 0,0 | 0,0 | 0,0 | 0,3 | 0,5 | 0,4 |
| reconstruit certificat (9) | 56,7 | 56,8 | 57,4 | 55,0 | 63,1 | 54,7 | 60,0 | 60,2 | 52,5 |
| vechi (2) | 0,9 | 0,9 | 2,5 | 2,7 | 2,5 | 2,7 | 5,7 | 6,5 | 7,3 |

**Perioada fără ancore (2025-07-01 – 2026-04-10)**

| Stare | time_1m | time_5m | volume_50btc | volume_adapt60s | dollar_5musdt | dollar_adapt60s | cusum_5bps | cusum_10bps | cusum_adapt_c1 |
|---|---|---|---|---|---|---|---|---|---|
| altele | 0,0 | 0,0 | 0,0 | 0,0 | 0,0 | 0,0 | 0,1 | 0,1 | 0,1 |
| fără book (1) | 3,0 | 3,0 | 2,8 | 2,8 | 2,1 | 2,9 | 2,3 | 2,1 | 2,6 |
| necertificat (10) | 0,0 | – | 0,0 | 0,0 | 0,0 | 0,0 | 0,5 | 0,8 | 0,7 |
| reconstruit certificat (9) | 96,5 | 96,5 | 95,1 | 95,0 | 95,8 | 95,0 | 90,6 | 89,6 | 88,0 |
| vechi (2) | 0,6 | 0,5 | 2,1 | 2,1 | 2,1 | 2,1 | 6,6 | 7,5 | 8,6 |

**Crash, 9–11 octombrie 2025**

| Stare | time_1m | time_5m | volume_50btc | volume_adapt60s | dollar_5musdt | dollar_adapt60s | cusum_5bps | cusum_10bps | cusum_adapt_c1 |
|---|---|---|---|---|---|---|---|---|---|
| necertificat (10) | – | – | 0,0 | 0,0 | 0,0 | 0,0 | 0,0 | 0,1 | 0,0 |
| reconstruit certificat (9) | 96,1 | 96,4 | 79,0 | 78,9 | 79,8 | 79,2 | 24,7 | 21,5 | 19,6 |
| vechi (2) | 3,9 | 3,6 | 20,9 | 21,1 | 20,2 | 20,8 | 75,3 | 78,4 | 80,4 |

**Crash, doar 10 octombrie 2025**

| Stare | time_1m | time_5m | volume_50btc | volume_adapt60s | dollar_5musdt | dollar_adapt60s | cusum_5bps | cusum_10bps | cusum_adapt_c1 |
|---|---|---|---|---|---|---|---|---|---|
| necertificat (10) | – | – | 0,1 | 0,0 | 0,0 | 0,0 | 0,0 | 0,1 | 0,0 |
| reconstruit certificat (9) | 88,6 | 89,6 | 56,9 | 57,0 | 58,3 | 57,8 | 10,4 | 8,8 | 10,6 |
| vechi (2) | 11,4 | 10,4 | 43,0 | 42,9 | 41,7 | 42,2 | 89,6 | 91,1 | 89,4 |

## Acoperire pe familii

**Tot istoricul**

| Familie | time_1m | time_5m | volume_50btc | volume_adapt60s | dollar_5musdt | dollar_adapt60s | cusum_5bps | cusum_10bps | cusum_adapt_c1 |
|---|---|---|---|---|---|---|---|---|---|
| Preț și traiectorie (F1) | 97,3 | 97,3 | 95,2 | 95,2 | 95,3 | 95,2 | 93,6 | 93,1 | 92,8 |
| Trades vs. index | 98,1 | 98,7 | 96,5 | 96,3 | 96,7 | 96,3 | 93,6 | 92,5 | 92,2 |
| Bară | 99,9 | 99,9 | 99,9 | 99,9 | 99,9 | 99,9 | 99,0 | 99,0 | 98,9 |
| Ticker, calendar (F10) | 99,4 | 99,5 | 99,0 | 99,0 | 98,9 | 99,0 | 98,1 | 97,9 | 97,8 |
| Volatilitate (F2) | 83,2 | 83,2 | 80,2 | 80,4 | 80,5 | 80,4 | 78,3 | 77,9 | 77,7 |
| Activitate și flux (F3) | 93,2 | 93,2 | 91,0 | 91,0 | 91,1 | 91,0 | 89,1 | 88,6 | 88,3 |
| Book, stare (F4) | 95,3 | 95,4 | 92,8 | 92,5 | 93,0 | 92,5 | 88,6 | 87,6 | 86,5 |
| Book, ferestre și OFI (F5) | 93,6 | 93,8 | 89,9 | 89,5 | 90,0 | 89,5 | 84,7 | 83,3 | 82,0 |
| Execuție (F6) | 93,2 | 93,3 | 88,7 | 88,5 | 89,0 | 88,6 | 83,3 | 82,3 | 81,8 |
| Book + flux (F6/F11) | 94,2 | 94,3 | 90,8 | 90,6 | 91,1 | 90,6 | 87,1 | 86,2 | 85,3 |
| Mark, basis, funding (F7) | 92,2 | 92,6 | 89,9 | 89,7 | 90,2 | 89,7 | 87,5 | 86,8 | 86,1 |
| Combinații funding (F7/F12) | 98,7 | 99,2 | 97,5 | 97,4 | 97,6 | 97,4 | 94,8 | 93,7 | 93,6 |
| Open interest (F8) | 99,5 | 99,8 | 99,0 | 99,0 | 99,1 | 98,9 | 99,0 | 99,1 | 98,9 |
| Lichidări (F9) | 96,7 | 96,7 | 96,1 | 96,2 | 96,1 | 96,2 | 95,0 | 95,0 | 94,8 |

**Perioada fără ancore (2025-07-01 – 2026-04-10)**

| Familie | time_1m | time_5m | volume_50btc | volume_adapt60s | dollar_5musdt | dollar_adapt60s | cusum_5bps | cusum_10bps | cusum_adapt_c1 |
|---|---|---|---|---|---|---|---|---|---|
| Preț și traiectorie (F1) | 97,3 | 97,3 | 95,2 | 95,4 | 95,3 | 95,4 | 92,5 | 92,0 | 91,5 |
| Trades vs. index | 99,4 | 99,5 | 98,1 | 98,0 | 97,9 | 98,0 | 94,6 | 93,5 | 92,7 |
| Bară | 100,0 | 99,9 | 99,9 | 99,9 | 99,9 | 99,9 | 99,0 | 99,0 | 98,9 |
| Ticker, calendar (F10) | 99,5 | 99,6 | 99,0 | 99,1 | 99,0 | 99,1 | 97,8 | 97,4 | 97,1 |
| Volatilitate (F2) | 83,8 | 83,8 | 80,3 | 80,9 | 80,7 | 80,9 | 77,3 | 76,8 | 76,5 |
| Activitate și flux (F3) | 93,4 | 93,5 | 91,1 | 91,3 | 91,2 | 91,3 | 88,1 | 87,6 | 87,0 |
| Book, stare (F4) | 94,8 | 94,9 | 92,1 | 92,0 | 92,6 | 92,0 | 86,8 | 85,6 | 83,6 |
| Book, ferestre și OFI (F5) | 93,1 | 93,2 | 88,9 | 88,8 | 89,4 | 88,8 | 82,6 | 81,3 | 78,7 |
| Execuție (F6) | 93,0 | 93,1 | 87,9 | 88,2 | 88,8 | 88,2 | 81,1 | 80,1 | 78,9 |
| Book + flux (F6/F11) | 94,1 | 94,2 | 90,6 | 90,8 | 91,2 | 90,8 | 85,6 | 84,7 | 83,3 |
| Mark, basis, funding (F7) | 94,3 | 94,4 | 91,7 | 91,6 | 91,7 | 91,7 | 88,5 | 87,8 | 86,7 |
| Combinații funding (F7/F12) | 99,7 | 99,7 | 98,6 | 98,6 | 98,5 | 98,6 | 95,5 | 94,4 | 93,8 |
| Open interest (F8) | 99,9 | 100,0 | 99,5 | 99,5 | 99,5 | 99,5 | 99,4 | 99,5 | 99,4 |
| Lichidări (F9) | 96,4 | 96,4 | 95,5 | 95,8 | 95,8 | 95,8 | 94,2 | 94,2 | 93,9 |

**Crash, 9–11 octombrie 2025**

| Familie | time_1m | time_5m | volume_50btc | volume_adapt60s | dollar_5musdt | dollar_adapt60s | cusum_5bps | cusum_10bps | cusum_adapt_c1 |
|---|---|---|---|---|---|---|---|---|---|
| Preț și traiectorie (F1) | 93,8 | 93,7 | 81,2 | 80,9 | 81,7 | 81,1 | 45,7 | 43,3 | 42,2 |
| Trades vs. index | 96,2 | 96,5 | 85,3 | 85,1 | 85,7 | 85,4 | 51,1 | 45,5 | 49,2 |
| Bară | 99,8 | 99,7 | 99,7 | 99,7 | 99,7 | 99,6 | 98,6 | 98,3 | 98,5 |
| Ticker, calendar (F10) | 98,3 | 98,5 | 93,8 | 93,8 | 94,1 | 93,9 | 77,7 | 75,3 | 76,8 |
| Volatilitate (F2) | 77,6 | 77,6 | 64,3 | 64,1 | 64,8 | 64,3 | 26,5 | 24,4 | 22,9 |
| Activitate și flux (F3) | 88,9 | 88,9 | 75,7 | 75,5 | 76,2 | 75,7 | 35,8 | 33,5 | 32,0 |
| Book, stare (F4) | 92,8 | 93,0 | 74,5 | 74,3 | 75,2 | 74,6 | 26,9 | 24,2 | 22,2 |
| Book, ferestre și OFI (F5) | 89,7 | 89,9 | 68,5 | 68,2 | 69,1 | 68,5 | 20,3 | 17,8 | 15,4 |
| Execuție (F6) | 90,1 | 90,4 | 68,7 | 68,5 | 69,4 | 68,8 | 20,2 | 17,6 | 15,5 |
| Book + flux (F6/F11) | 90,4 | 90,6 | 72,1 | 71,9 | 72,7 | 72,2 | 28,4 | 25,9 | 24,3 |
| Mark, basis, funding (F7) | 88,4 | 88,7 | 77,9 | 77,8 | 78,3 | 78,0 | 52,6 | 50,0 | 51,1 |
| Combinații funding (F7/F12) | 97,5 | 97,8 | 88,6 | 88,5 | 89,0 | 88,7 | 59,7 | 53,1 | 58,3 |
| Open interest (F8) | 99,9 | 100,0 | 99,7 | 99,7 | 99,7 | 99,7 | 99,9 | 99,9 | 99,9 |
| Lichidări (F9) | 93,9 | 94,0 | 87,9 | 87,9 | 88,2 | 88,0 | 77,0 | 77,1 | 76,4 |

**Crash, doar 10 octombrie 2025**

| Familie | time_1m | time_5m | volume_50btc | volume_adapt60s | dollar_5musdt | dollar_adapt60s | cusum_5bps | cusum_10bps | cusum_adapt_c1 |
|---|---|---|---|---|---|---|---|---|---|
| Preț și traiectorie (F1) | 83,9 | 83,8 | 62,4 | 62,3 | 63,4 | 62,7 | 35,7 | 34,4 | 35,9 |
| Trades vs. index | 93,7 | 93,5 | 75,0 | 75,0 | 75,7 | 75,3 | 43,3 | 38,1 | 44,1 |
| Bară | 99,5 | 99,2 | 99,3 | 99,3 | 99,4 | 99,3 | 98,4 | 98,1 | 98,4 |
| Ticker, calendar (F10) | 96,5 | 96,5 | 88,9 | 88,9 | 89,3 | 89,1 | 73,9 | 71,8 | 74,3 |
| Volatilitate (F2) | 66,1 | 66,1 | 44,2 | 44,2 | 45,1 | 44,6 | 15,9 | 15,1 | 16,1 |
| Activitate și flux (F3) | 79,4 | 79,4 | 56,5 | 56,4 | 57,5 | 56,9 | 24,9 | 23,9 | 25,1 |
| Book, stare (F4) | 83,4 | 83,9 | 52,1 | 52,1 | 53,3 | 52,8 | 13,9 | 12,7 | 14,0 |
| Book, ferestre și OFI (F5) | 77,9 | 78,2 | 42,9 | 42,9 | 44,0 | 43,6 | 6,7 | 5,7 | 6,8 |
| Execuție (F6) | 80,7 | 81,2 | 45,6 | 45,6 | 46,8 | 46,4 | 7,3 | 6,0 | 7,4 |
| Book + flux (F6/F11) | 79,4 | 79,6 | 49,6 | 49,5 | 50,6 | 50,1 | 16,5 | 15,4 | 16,6 |
| Mark, basis, funding (F7) | 85,9 | 86,1 | 69,0 | 69,0 | 69,6 | 69,3 | 46,8 | 44,6 | 47,1 |
| Combinații funding (F7/F12) | 95,7 | 95,5 | 80,1 | 80,1 | 80,8 | 80,4 | 53,0 | 46,5 | 53,9 |
| Open interest (F8) | 100,0 | 100,0 | 99,7 | 99,6 | 99,7 | 99,6 | 99,9 | 99,9 | 99,9 |
| Lichidări (F9) | 88,6 | 88,6 | 80,5 | 80,5 | 80,8 | 80,6 | 74,3 | 74,9 | 74,3 |

## Features reprezentative

**Tot istoricul**

| Feature | time_1m | time_5m | volume_50btc | volume_adapt60s | dollar_5musdt | dollar_adapt60s | cusum_5bps | cusum_10bps | cusum_adapt_c1 |
|---|---|---|---|---|---|---|---|---|---|
| add_bid_10bps_2m_btc | 90,2 | 90,5 | 83,9 | 83,5 | 83,7 | 83,5 | 78,4 | 77,3 | 75,4 |
| basis_mark_index_bps | 98,7 | 99,2 | 97,5 | 97,4 | 97,6 | 97,4 | 94,8 | 93,7 | 93,6 |
| cost_buy_1btc_bps | 97,3 | 97,3 | 95,8 | 95,6 | 96,1 | 95,6 | 92,4 | 91,5 | 90,6 |
| depth_bid_10bps_btc | 96,9 | 96,9 | 95,4 | 95,1 | 95,6 | 95,1 | 91,6 | 90,6 | 89,6 |
| depth_bid_25bps_btc | 94,9 | 95,0 | 92,8 | 92,4 | 92,9 | 92,4 | 89,0 | 88,1 | 86,8 |
| depth_bid_50bps_btc | 88,1 | 88,1 | 85,9 | 85,4 | 85,6 | 85,5 | 82,9 | 82,3 | 80,4 |
| depth_bid_5bps_btc | 97,3 | 97,3 | 95,8 | 95,5 | 96,0 | 95,6 | 92,1 | 91,2 | 90,3 |
| depth_bid_top10_btc | 97,3 | 97,3 | 95,8 | 95,6 | 96,1 | 95,6 | 92,2 | 91,3 | 90,4 |
| eff_spread_qw_bps_2m | 94,4 | 94,7 | 90,7 | 90,2 | 90,9 | 90,2 | 85,5 | 84,2 | 82,7 |
| funding_rate_asof | 99,0 | 99,3 | 98,3 | 98,2 | 98,3 | 98,1 | 96,4 | 95,7 | 95,5 |
| liq_count_15m | 97,1 | 97,1 | 96,2 | 96,3 | 96,2 | 96,3 | 95,0 | 95,0 | 94,5 |
| liq_count_1h | 91,6 | 91,6 | 89,0 | 89,7 | 89,2 | 89,7 | 86,0 | 86,2 | 85,9 |
| markout_30s_qw_bps_15m | 85,2 | 85,2 | 74,2 | 74,7 | 75,0 | 74,8 | 65,7 | 65,0 | 66,6 |
| mlofi_l5_30s_btc | 96,3 | 96,5 | 93,5 | 93,1 | 93,8 | 93,1 | 88,0 | 86,4 | 85,4 |
| ofi_15m_btc | 86,4 | 86,4 | 80,0 | 79,6 | 80,1 | 79,6 | 75,1 | 74,2 | 72,6 |
| ofi_2m_btc | 94,4 | 94,7 | 90,7 | 90,2 | 90,9 | 90,3 | 85,5 | 84,2 | 82,7 |
| ofi_30s_btc | 96,3 | 96,5 | 93,5 | 93,2 | 93,8 | 93,2 | 88,3 | 86,9 | 85,7 |
| oi_btc_asof | 99,4 | 99,9 | 98,6 | 98,5 | 98,7 | 98,5 | 98,6 | 98,8 | 98,5 |
| qi_1 | 97,3 | 97,3 | 95,8 | 95,6 | 96,1 | 95,6 | 92,4 | 91,6 | 90,7 |
| ret_trade_1h | 99,8 | 99,8 | 99,7 | 99,7 | 99,6 | 99,7 | 99,6 | 99,2 | 99,6 |
| ret_trade_24h | 99,6 | 99,6 | 99,6 | 99,7 | 99,6 | 99,7 | 99,7 | 99,3 | 99,6 |
| rv_trade_1h_grid1s | 88,7 | 88,7 | 82,4 | 82,6 | 82,9 | 82,7 | 80,3 | 79,9 | 79,4 |
| spread_bps | 97,3 | 97,3 | 95,8 | 95,6 | 96,1 | 95,6 | 92,4 | 91,6 | 90,7 |
| tfi_1h_btc | 88,7 | 88,7 | 82,4 | 82,6 | 82,9 | 82,7 | 80,3 | 79,9 | 79,4 |
| tw_spread_bps_2m | 94,4 | 94,7 | 90,7 | 90,2 | 90,9 | 90,2 | 85,4 | 84,0 | 82,5 |

**Perioada fără ancore (2025-07-01 – 2026-04-10)**

| Feature | time_1m | time_5m | volume_50btc | volume_adapt60s | dollar_5musdt | dollar_adapt60s | cusum_5bps | cusum_10bps | cusum_adapt_c1 |
|---|---|---|---|---|---|---|---|---|---|
| add_bid_10bps_2m_btc | 89,6 | 90,1 | 82,1 | 82,0 | 82,4 | 81,9 | 75,9 | 74,8 | 71,4 |
| basis_mark_index_bps | 99,7 | 99,7 | 98,6 | 98,6 | 98,5 | 98,6 | 95,5 | 94,4 | 93,8 |
| cost_buy_1btc_bps | 96,5 | 96,5 | 95,1 | 95,0 | 95,8 | 95,0 | 90,5 | 89,5 | 87,9 |
| depth_bid_10bps_btc | 96,2 | 96,3 | 94,7 | 94,6 | 95,4 | 94,6 | 89,6 | 88,4 | 86,7 |
| depth_bid_25bps_btc | 94,3 | 94,4 | 91,9 | 91,7 | 92,4 | 91,7 | 86,7 | 85,6 | 83,4 |
| depth_bid_50bps_btc | 86,9 | 86,9 | 83,6 | 83,3 | 83,8 | 83,3 | 79,6 | 78,8 | 75,9 |
| depth_bid_5bps_btc | 96,4 | 96,5 | 95,0 | 94,9 | 95,7 | 94,9 | 90,2 | 89,0 | 87,5 |
| depth_bid_top10_btc | 96,5 | 96,5 | 95,1 | 95,0 | 95,7 | 95,0 | 90,3 | 89,1 | 87,5 |
| eff_spread_qw_bps_2m | 94,6 | 95,0 | 90,5 | 90,4 | 91,0 | 90,4 | 83,7 | 82,4 | 79,9 |
| funding_rate_asof | 99,8 | 99,8 | 99,1 | 99,1 | 99,0 | 99,1 | 96,8 | 96,1 | 95,5 |
| liq_count_15m | 96,9 | 96,9 | 95,7 | 96,0 | 96,0 | 96,0 | 94,3 | 94,3 | 94,1 |
| liq_count_1h | 89,7 | 89,8 | 86,5 | 87,6 | 87,5 | 87,7 | 83,0 | 83,3 | 82,7 |
| markout_30s_qw_bps_15m | 85,4 | 85,5 | 72,7 | 74,2 | 74,4 | 74,5 | 62,4 | 62,0 | 63,7 |
| mlofi_l5_30s_btc | 95,9 | 96,1 | 93,2 | 93,1 | 93,8 | 93,1 | 86,5 | 84,9 | 82,6 |
| ofi_15m_btc | 86,8 | 86,8 | 79,0 | 79,2 | 79,6 | 79,2 | 72,6 | 71,9 | 69,4 |
| ofi_2m_btc | 94,6 | 95,0 | 90,5 | 90,4 | 91,0 | 90,4 | 83,8 | 82,4 | 79,9 |
| ofi_30s_btc | 95,9 | 96,1 | 93,3 | 93,1 | 93,8 | 93,1 | 86,6 | 85,1 | 82,8 |
| oi_btc_asof | 99,8 | 100,0 | 99,3 | 99,3 | 99,3 | 99,3 | 99,1 | 99,2 | 99,2 |
| qi_1 | 96,5 | 96,5 | 95,1 | 95,0 | 95,8 | 95,0 | 90,6 | 89,6 | 88,0 |
| ret_trade_1h | 99,8 | 99,8 | 99,6 | 99,6 | 99,6 | 99,6 | 99,6 | 99,2 | 99,5 |
| ret_trade_24h | 99,8 | 99,8 | 99,6 | 99,6 | 99,6 | 99,6 | 99,6 | 99,2 | 99,5 |
| rv_trade_1h_grid1s | 88,6 | 88,6 | 81,8 | 82,8 | 82,6 | 82,9 | 78,0 | 77,7 | 77,8 |
| spread_bps | 96,5 | 96,5 | 95,1 | 95,0 | 95,8 | 95,0 | 90,6 | 89,6 | 88,0 |
| tfi_1h_btc | 88,6 | 88,6 | 81,8 | 82,8 | 82,6 | 82,9 | 78,0 | 77,7 | 77,8 |
| tw_spread_bps_2m | 94,6 | 95,0 | 90,5 | 90,4 | 91,0 | 90,4 | 83,8 | 82,4 | 79,9 |

**Crash, 9–11 octombrie 2025**

| Feature | time_1m | time_5m | volume_50btc | volume_adapt60s | dollar_5musdt | dollar_adapt60s | cusum_5bps | cusum_10bps | cusum_adapt_c1 |
|---|---|---|---|---|---|---|---|---|---|
| add_bid_10bps_2m_btc | 82,8 | 83,2 | 57,7 | 57,4 | 58,2 | 57,5 | 16,5 | 14,7 | 11,7 |
| basis_mark_index_bps | 97,5 | 97,8 | 88,6 | 88,5 | 89,0 | 88,7 | 59,7 | 53,1 | 58,3 |
| cost_buy_1btc_bps | 96,1 | 96,4 | 79,0 | 78,9 | 79,8 | 79,2 | 24,7 | 21,5 | 19,5 |
| depth_bid_10bps_btc | 95,6 | 95,9 | 78,5 | 78,3 | 79,2 | 78,7 | 24,5 | 21,3 | 19,4 |
| depth_bid_25bps_btc | 91,6 | 91,8 | 72,9 | 72,7 | 73,6 | 72,9 | 22,2 | 19,9 | 17,1 |
| depth_bid_50bps_btc | 81,8 | 81,7 | 59,5 | 59,1 | 60,0 | 59,4 | 17,7 | 16,3 | 12,8 |
| depth_bid_5bps_btc | 96,0 | 96,3 | 79,0 | 78,8 | 79,8 | 79,2 | 24,7 | 21,5 | 19,5 |
| depth_bid_top10_btc | 96,1 | 96,4 | 79,0 | 78,9 | 79,8 | 79,2 | 24,7 | 21,5 | 19,5 |
| eff_spread_qw_bps_2m | 91,4 | 91,9 | 68,2 | 68,0 | 68,9 | 68,3 | 19,5 | 16,8 | 14,7 |
| funding_rate_asof | 98,0 | 98,3 | 92,3 | 92,3 | 92,7 | 92,5 | 69,2 | 65,5 | 68,1 |
| liq_count_15m | 94,5 | 94,6 | 87,7 | 87,7 | 88,0 | 87,8 | 80,0 | 78,7 | 79,8 |
| liq_count_1h | 81,6 | 81,6 | 64,6 | 64,5 | 65,4 | 65,0 | 23,5 | 24,7 | 20,9 |
| markout_30s_qw_bps_15m | 79,1 | 79,4 | 53,5 | 53,3 | 54,1 | 53,5 | 14,5 | 12,9 | 10,7 |
| mlofi_l5_30s_btc | 94,4 | 94,8 | 73,9 | 73,6 | 74,6 | 73,9 | 21,9 | 19,0 | 16,7 |
| ofi_15m_btc | 79,6 | 79,9 | 54,1 | 53,9 | 54,7 | 54,2 | 14,7 | 13,0 | 10,8 |
| ofi_2m_btc | 91,4 | 91,9 | 68,3 | 68,0 | 69,0 | 68,3 | 19,5 | 16,8 | 14,7 |
| ofi_30s_btc | 94,4 | 94,8 | 73,9 | 73,6 | 74,6 | 73,9 | 21,9 | 19,0 | 16,7 |
| oi_btc_asof | 99,8 | 100,0 | 99,6 | 99,5 | 99,6 | 99,5 | 99,8 | 99,8 | 99,8 |
| qi_1 | 96,1 | 96,4 | 79,0 | 78,9 | 79,8 | 79,2 | 24,7 | 21,5 | 19,6 |
| ret_trade_1h | 99,1 | 99,0 | 95,8 | 95,6 | 96,2 | 95,5 | 97,7 | 95,8 | 97,7 |
| ret_trade_24h | 98,9 | 98,6 | 96,1 | 96,0 | 96,6 | 95,9 | 98,8 | 96,8 | 98,9 |
| rv_trade_1h_grid1s | 79,6 | 79,6 | 59,3 | 58,9 | 59,8 | 59,1 | 18,2 | 15,9 | 13,6 |
| spread_bps | 96,1 | 96,4 | 79,0 | 78,9 | 79,8 | 79,2 | 24,7 | 21,5 | 19,6 |
| tfi_1h_btc | 79,6 | 79,6 | 59,3 | 58,9 | 59,8 | 59,1 | 18,2 | 15,9 | 13,6 |
| tw_spread_bps_2m | 91,4 | 91,9 | 68,3 | 68,0 | 69,0 | 68,3 | 19,5 | 16,8 | 14,7 |

## Metadate de calitate

Coloane păstrate în fișiere, excluse implicit din features pentru model (`chdds.dataset.model_columns`, `model_feature_columns` în manifest): `book_state`, `cert_bid_floor_dist_bps`, `cert_ask_ceiling_dist_bps`.

Incidente pentru excludere în experimente (`quality_incidents.json`, `chdds.dataset.incident_rows`). Nu sunt coloane în dataset și nu se folosesc ca input:

- `book_phantom_level_2025-08-29`: 2025-08-29 06:37:05+00:00 – 2025-08-29 06:37:36+00:00. Before the first crossed message of the stream, a level later shown inconsistent can appear as certified top of book. Not detectable causally.

Bare care ating intervalul: time_1m 1, time_5m 1, volume_50btc 4, volume_adapt60s 4, dollar_5musdt 3, dollar_adapt60s 4, cusum_5bps 53, cusum_10bps 1, cusum_adapt_c1 51.
