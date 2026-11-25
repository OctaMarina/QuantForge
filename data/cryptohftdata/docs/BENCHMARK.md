# Benchmark (măsurat)

Mașină: 22 nuclee, 30 GB RAM (≈ 13–20 GB disponibili în timpul rulărilor; `systemd-oomd` activ), NVMe.
Memorie = „Maximum resident set size” raportat de `/usr/bin/time -v` (procesul principal; în etapele paralele
fiecare worker are în plus ~1,5–3 GB). Timpi wall-clock.

## 1. Intervale reprezentative (înaintea rulării extinse)

Configurație: 4 seturi de bare (time 1 m, volume 50 BTC, dollar 5 M USDT, CUSUM 5 bps), toate familiile, 4 workers.

| Interval | Particularități | canonical | bars | features | book | assemble | ieșire dataset |
|---|---|---|---|---|---|---|---|
| 2026-04-22 → 25 (4 zile) | întreruperea mark/ticker 24 h, lichidări 23 h, puține ancore | 19 s / 1,1 GB | 27 s / 3,7 GB | 75 s / 3,2 GB | 216 s / 2,4 GB (149 s cu regula de frontieră și mașina mai puțin încărcată) | 10 s / 0,4 GB | 64,5 MB |
| 2026-05-14 → 16 (3 zile) | multe goluri și re-ancorări, mesaje la 26 ms | 11 s / 0,9 GB | 12 s / 3,0 GB | 33 s / 2,4 GB | 112 s / 2,6 GB | 8 s / 0,4 GB | 54,5 MB |
| 2025-06-28 → 30 (demo, 6 seturi) | 3 copii/eveniment, snapshot-uri sintetice | — | 19 s | 88 s (1 worker) | 133–185 s (3 workers) | 14 s | 194 MB (cu bare de 5 s) |

Incident în prima rulare completă: `free(): invalid pointer` pe 2026-09-21 — checkpoint-urile furnizorului (din
2026-08-22) conțin ask-uri până la 4,5 mil. USDT, în afara grilei de tick-uri; scrierile din ramura de verificare nu
verificau limitele. Corectat + test; replay-ul a fost reluat de la zero (nu din checkpoint), pentru că o scriere în
afara limitelor putea corupe silențios starea din aug.–sept. 2026.

Concluzii aplicate înainte de rularea completă: (1) checkpoint-urile replay-ului erau ~310 MB/zi necomprimate →
`savez_compressed` + retenție doar sfârșit de lună și ultimul; (2) motoarele de bare prealocau ieșirea proporțional
cu numărul de tranzacții (≈1 GB/proces) → `systemd-oomd` a oprit prima rulare → procesare în bucăți de 200 k
tranzacții (echivalență testată) și paralelism redus; (3) orele fără snapshot în care book-ul este neancorat nu
mai sunt decodate (rezultat identic, testat) — perioada iul. 2025 → apr. 2026 trece în ~5 s/zi.

## 2. Rularea completă (2025-06-28 → 2026-10-31, 9 seturi de bare)

| Etapă | Wall | Max RSS proces principal | Paralelism | Note |
|---|---|---|---|---|
| audit (.zst, 59.769 fișiere) | 73 min | ~1,5 GB/worker | 9 | 50,79 mld. rânduri |
| audit (.parquet, 10.585 fișiere) | 18 min | ~1,5–3 GB/worker | 9 | 9,72 mld. rânduri |
| canonical | 21 min 50 s | 3,7 GB | 1 | 1,77 mld. tranzacții economice |
| bars (9 seturi) | 15 min 48 s | 9,6 GB | 3 | 9.413.305 bare |
| features | 2 h 02 min 32 s | 9,8 GB | 4 | 9.413.305 rânduri × ~390 coloane |
| book (replay L2 unic pentru toate seturile; rularea finală, după corecția integrării) | 1 h 35 min 33 s | 7,6 GB | 4 decodări | 8.227 ore neancorate fără snapshot sărite (exact); 3.557 ore decodate; 391,97 mil. mesaje, 16,4 mld. niveluri aplicate; ~9,4 mil. interogări (timpi de evaluare unici) |
| split pe seturi + statistici | 2 min 55 s | 1,3 GB | 1 | |
| recalcul features CUSUM adaptiv (după corecția motorului) | ~47 min | | 4 | |
| assemble (9 seturi, 9.413.305 rânduri × 664–670 coloane) | 13 min 48 s | 4,2 GB | 1 | 23 GB parquet zstd |
| demo 2025-06-28 → 30 (6 seturi, toate etapele) | 2 min 14 s | 2,8 GB | 2 | 203 MB |
| verificare independentă (5 zile × 9 seturi) | 7 min 45 s | 2,7 GB | 1 | |

Spațiu pe disc (derived, ~99 GB): staged 28 GB, canonical 17 GB, bars 1,2 GB, features 18 GB, book_union 3,1 GB,
book_features 3,9 GB, book_grid (100 ms) 3,6 GB, book_state (checkpoint-uri lunare) 0,8 GB, datasets 23 GB, demo 0,2 GB.
`features/`, `book_union/` și `book_features/` sunt intermediare (necesare pentru extindere incrementală fără recalcul);
pot fi șterse după asamblare dacă spațiul este critic (disc liber după rulare: 81 GB).
Total pipeline de la zero (audit → dataset) ≈ 6 h pe această mașină.
