# Limitări, blocaje și decizii încă necesare

Stare la 2026-11-03. Fiecare punct indică dovezile (fișiere/tabele) și opțiunile. Nimic de aici nu este „rezolvat”
prin aproximări prezentate drept exacte.

## Decizii deschise (necesită alegerea utilizatorului)

### D1. Book-ul L2 în perioada fără ancore (iul. 2025 → 10.04.2026)
- **Stare:** investigată integral, vezi `docs/D1_REPORT.md`. Așteaptă decizia de regenerare.
- **Dovezi.** 0 snapshot-uri între 2025-07-01 și 2026-04-10 (absență reală). Rezolverul vechi pierdea în plus ancore existente
  în aprilie–octombrie 2026. Cu el au fost construite dataseturile actuale (valid 26,2%). Cu rezolverul corectat: 40,7%.
- **Implementat:** rezolverul corectat. `btcusdt_v2` e regenerat cu el; `btcusdt_initial` rămâne neschimbat.
- **Implementat, opțional:** `book_mode: rebuilt`. Fără ancoră, book-ul pornește gol și emite doar sub un certificat exact derivat
  din protocol (`book_state` 9). Măsurat în `btcusdt_v2`: vârf de book 97,3% pe bare de 1 minut (`docs/DATASET_V2.md`).
- **Rămâne incert:** exactitatea în cele 9,3 luni e dedusă, nu măsurată direct. Episodul de flux încrucișat din 2025-08-29 are
  cauză necunoscută. Acoperirea pe bare bazate pe evenimente scade mult în crash (CUSUM 5 bps: 24,7% pe 9–11.10.2025).
- **Opțiuni:** (a) regenerezi strict cu rezolverul corectat; (b) adaugi un dataset separat în modul opțional; (c) ceri furnizorului
  stadiul backfill-ului de snapshot-uri orare pentru această perioadă.
- `scripts/heal_experiment.py` (continuarea book-ului vechi peste goluri) nu testează metoda furnizorului și e înlocuit de
  `scripts/d1_experiment.py`.

### D2. Open interest 15.10.2026 (rezolvat, cu 8 puncte în carantină)
- **Cauză: datele, nu interpretarea noastră.** Pe 2026-10-15 10:55–14:45 sursa a publicat 47 de puncte de 5 min cu versiuni
  contradictorii: valori înghețate, o sursă cu valoare 0 și puncte în afara grilei, și o serie live imposibilă fizic (salt de
  8.149 BTC într-un interval cu 752 BTC tranzacționați). Eroarea noastră era politica „prima versiune”, care folosea tocmai acele
  valori, plus lipsa carantinei pentru rândurile cu valoare 0 căzute pe grilă.
- **Implementat:** o versiune e respinsă numai dacă e demonstrabil imposibilă (`canonical/slow.py`, `classify_oi_versions`). 39 de
  puncte se rezolvă cu singura versiune rămasă, disponibilă abia de la recepție (16 h sau mai târziu). **8 puncte rămân în carantină**
  (10:55, 12:30, 12:35, 13:20–13:40): au cel puțin două versiuni plauzibile și nu se alege între ele.
- **Propagare:** valorile respinse nu intră în nicio feature; referința expiră, iar `oi_state=3` arată motivul; statisticile rolling
  folosesc doar punctele deja primite. Observațiile cu o singură versiune rămân neatinse, cu excepția unui rând cu valoare 0.
- Respingerea poate acționa retroactiv pentru că doar reține o valoare; nicio valoare nu apare înainte de recepție.

### D3. Ore fără fișier de lichidări (75 de ore)
- **Decis:** rămân necunoscute. `liq_state=2`, features `null`, fără transformare în zero. `time_since_*_liq_s` e `null` dacă o oră
  fără fișier sau un gol de socket se află după ultima notificare. Politica `inferred_zero` rămâne doar opțională și marcată.
- `liq_state=3` înseamnă gol pe socket-ul comun: nici mark, nici ticker. Un gol doar în mark nu mai invalidează lichidările.

### D4. Ferestre lungi și golurile de tranzacții
- **Decis pentru prima versiune:** statisticile de flux și traiectorie rămân stricte (coverage 100%), fără variante parțiale. La 1 min,
  fracția validă e 88,74% pe 1 h, 66,96% pe 4 h și 15,02% pe 24 h. Cauza: 2.846 goluri de ID.
- `ret_trade_*` e definit doar de două prețuri, deci nu mai e respins de golurile din interior; rămâne `null` dacă un gol cade între
  un preț și capătul lui, sau dacă un capăt e mai vechi de 10 s.

## Limitări structurale (nu se pot recupera din aceste surse)
- Identități de conturi/ordine, levier, prioritate în coadă, lichiditate ascunsă, lichidări neraportate (feed-ul
  Binance publică cel mult un ordin de lichidare/simbol/1000 ms), arbitraj între burse.
- `order_count` este integral null; Premium Index oficial nu poate fi reconstruit (lipsesc parametri istorici).
- Lichidările nu au `order_id`: sumele sunt „notional notificat”, nu limită inferioară demonstrată.

## Limitări de implementare (inventar complet în `FEATURE_REGISTRY.md`)
- Neimplementate: λ (sensibilitate la flux), persistența basis, cuantile temporale ale stărilor, episoade de
  reziliență, Amihud, micro-price calibrat, Hawkes, VPIN-like, Premium Index, VAR, entropie, regimuri învățate,
  tensori L2, bare pe ceasul de piață, varianta cu fragmentarea tranzacțiilor, CUSUM pe mid/grilă.
- Features orientate de semnal (`s_k × …`) cer semnalul primar — etapă ulterioară (separare de ținte).
- Integralele temporale folosesc float64 rebazat orar (eroare relativă < 1e−9 pe ferestre ≥ 1 s; pragul de coverage
  strict este 1 − 1e−7).
- Prospețimea book-ului se măsoară pe vârsta de piață (`eval_time − event_time`), care include decalajul de ceas al
  colectorului (întârzieri negative observate: vezi auditul); pragul 1 s nu a fost ajustat pe rezultate.

### D5. Markouts (rezolvat)
Scadențele nu mai trec peste sfârșitul lotului orar, iar ferestrele cer acoperire pe `w + h` plus vârsta maximă a cotației. Se aplică
în ambele moduri. Confirmat prin recalculare independentă pe intervale nefolosite la corectare.
