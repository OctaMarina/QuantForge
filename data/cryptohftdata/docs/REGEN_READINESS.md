# Pregătirea pentru regenerare (2026-11-05)

Stare: regenerarea a fost executată pe 2026-11-05; rezultatele sunt în `docs/DATASET_V2.md`. Comanda: `bash scripts/run_regen.sh`, configurația
`configs/btcusdt_v2.yaml` (aceleași bare și ferestre ca `btcusdt_initial`, book reconstruit cu certificat). Dovezile sunt în
`derived/regen_check/report.json` și `dry_run.json`; verificările se reiau cu `scripts/verify_regen_ready.py`, rularea de probă cu
`scripts/run_regen_dry.sh`. Nicio scanare integrală nouă nu a fost necesară.

## Corecții

| Zonă | Corecție |
|---|---|
| Book | Rezolver corectat implicit; `book_mode: rebuilt` în configurație; `book_state` 0/9/10 pe fiecare rând și `book_provenance` plus stări pe lună în manifest. La încrucișare certificatul se anulează, iar după carantină book-ul repornește gol. |
| Markout | Scadențele nu mai trec peste sfârșitul lotului; ferestrele cer `w + h` plus vârsta cotației; baza inițială a ferestrei supraviețuiește rebase-ului. Se aplică și în strict, inclusiv la effective spread. |
| Open interest | Versiunile sunt clasificate în `canonical/slow.py`. 39 din 47 de puncte contestate se rezolvă, 8 stau în carantină. Statisticile rolling folosesc doar puncte deja primite. |
| Lichidări | `time_since_*_liq_s` nu mai presupune zero liniște peste ore fără fișier sau goluri de socket. `liq_state=3` cere tăcere pe mark și pe ticker. |
| Mark/funding | `funding_change` verifică vechimea capătului vechi; pauzele nu mai repetă rata în statistica robustă; intervalul de funding e necunoscut după o tăcere care ar putea ascunde un ciclu. |
| Ferestre lungi | `ret_trade_*` depinde doar de cele două prețuri și de golurile dintre preț și capăt. Statisticile de flux și traiectorie rămân stricte. |
| Subseturi | `split_book` acceptă bare evaluate înaintea începutului unui subset (book `null`). |

## Clarificări

- **Diferențele de rang:** artefact de rotunjire la milisecundă în pasul pe vârste (un update sosit cu 832 µs înainte de T). Book-urile sunt corecte.
- **OI 15.10.2026:** provine din date. Interpretarea noastră adăuga o eroare: politica „prima versiune” alegea tocmai seria imposibilă.
- **Mark:** golurile și vechimea erau deja tratate corect pentru valorile directe și pentru basis; omisiunile erau cele din tabel.

## Verificări

| Verificare | Rezultat |
|---|---|
| Teste | 74 trec |
| Markout, recalculare independentă, 2026-09-17 și 2026-06-14, nefolosite la corectare | 133.644 de comparații, 27.078 lângă granițe: 0 diferențe de număr, ≤ 4e−10 bps |
| Strict vs. valorile existente, 2025-06-28/29 | 9.013.898 de valori; schimbate doar 21 de coloane de markout și effective spread |
| Încrucișare reală, 2025-08-29 | 0 rânduri certificate între prima încrucișare și 07:02:24; apoi spread median de 1 tick |
| OI | 0 valori respinse folosite; un singur rând cu o singură versiune respins (valoare 0) |
| Lichidări, 40 din 75 de ore lipsă | toate necunoscute, 0 zerouri, `time_since` null |
| Mark, pană 2026-04-24 și goluri 2026-09-17 | 0 valori în afara limitelor de vechime; basis-ul din book e null cu mark vechi |
| Rulare cap-coadă, 5 ferestre, 81.115 bare | 0 erori de conservare, 0 erori brute-force pe 26.065 de recalculări, 0 valori sub măști, 0 coloane neînregistrate |
| Reluare din checkpoint în modul reconstruit | identică bit cu bit |

Efect măsurat pe ferestrele de probă, `time_1m`: `ret_trade_24h` de la 0–4% la peste 99%; `liq_count_1h` pe 16–17.09.2026 de la 32% la 95%.

## Limitări acceptate în dataset (tratate explicit)

- Vârful de book din iulie 2025 – 10 aprilie 2026 e exact prin certificat, dedus din protocol, fără referință directă.
- 2025-08-29: 31 de secunde (06:37:05–06:37:36) înaintea primei încrucișări pot avea un nivel problematic ca vârf certificat; cauza episodului e necunoscută. Carantina de 600 s e o măsură de prudență.
- OI: 8 puncte în carantină; valorile rezolvate apar abia la recepție, deci pe 15.10.2026 de la 11:10 referința expiră până la primul punct valid. Cele 2.768 de puncte în afara grilei rămân excluse.
- Lichidări: 75 de ore necunoscute. Tăcerea socket-ului e indiciu, nu dovadă, despre starea fluxului.
- Mark/funding: golurile rămân goluri; pana din aprilie 2026 lasă features `null`.
- Ferestre de flux și traiectorie lungi rămân rare (24 h: circa 15% la 1 min), prin decizie.
- Acoperirea scade în crash, mai ales pe bare de evenimente.

## Bug-uri deschise

Niciunul cunoscut.

## Resurse

Proiecție inițială: vârf de 43,8 GiB, 52,6 cu marjă. Rulare reală: datasetul 34 GB, features 18 GB, grila 4,5 GB, stare 1,2 GB; minimul de spațiu liber 16,4 GiB, peste limita de 10 GiB
(`derived/regen_check/btcusdt_v2/disk_monitor.csv`). Discul e la 99%. Rerularea continuă după oprire, verificat bit cu bit prin oprire în features, book și assemble.
