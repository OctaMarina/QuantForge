# D1 — features de order book pe intervalele fără ancoră

Investigație încheiată pe 2026-11-05. Tabelele complete sunt în `docs/D1_TABLES.md` (generate automat), datele în
`derived/d1/`. Istoricul a fost regenerat ulterior ca `btcusdt_v2`, iar acoperirile de mai jos sunt măsurate pe el (`docs/DATASET_V2.md`).

## Concluzie

Putem recupera justificat features de book pe aproape toată perioada fără ancore, fără warm-up de timp, cu un certificat
exact derivat din protocol. Pentru vârful de book, acoperirea măsurată pe barele de 1 minut urcă de la 26,2% la 97,3%.

| | Valid din istoric | Sursa |
|---|---|---|
| Strict, așa cum e acum în dataseturi | 26,2% | măsurat |
| Strict, cu rezolverul corectat | 40,7% | măsurat, tot istoricul |
| Strict corectat + book reconstruit certificat (vârf de book) | 97,3% | măsurat pe `btcusdt_v2`, `time_1m` |

Două lucruri stau la baza acestui rezultat, și sunt separate:

1. **O bună parte din lipsa de ancore era eroarea noastră.** În aprilie–octombrie 2026, snapshot-urile de la reconectare
   existau, dar rezolverul nu le folosea. Corecția ridică validitatea din iunie–septembrie 2026 de la 48–76% la peste 99,5%.
   Lunile iulie 2025 – 10 aprilie 2026 rămân fără nicio ancoră: acolo snapshot-urile lipsesc cu adevărat.
2. **Un book pornit gol poate dovedi singur că vârful e exact.** Pe 1,6 milioane de comparații cu referința nu a ținut
   nicio cantitate greșită și niciun nivel fals, iar valorile emise sub certificat au fost identice cu referința.

Limita principală: **în cele 9,3 luni fără ancore nu există referință**. Acolo exactitatea se sprijină pe protocol și pe
verificările indirecte din §7, nu pe o comparație directă.

Recomand certificatul, nu un warm-up de timp. Warm-up-ul de timp trece global, dar eșuează exact în orele calme și adânci,
iar timpul din articolul furnizorului nu se generalizează (§5).

## 1. Detectarea snapshot-urilor

Am verificat implementarea contra particularităților documentate de furnizor, pe toate cele 15.343 de snapshot-uri.

| Verificare | Rezultat |
|---|---|
| Snapshot-uri în iulie 2025 – 10 aprilie 2026 | **0**. Absența e reală, nu o eroare de grupare |
| Snapshot-uri scrise în două jumătăți cu `received_time` diferit | 0 pentru BTCUSDT. Cele 13.301 cu mai multe părți sunt copii identice |
| Identificare prin `event_time` + `last_update_id` | gruparea veche era corectă pe aceste date, dar ar fi greșit pe jumătăți reale. Corectat și testat |
| Mesajul care conectează snapshot-ul, primit înaintea lui | 4.727 de cazuri. **Defect A** |
| Snapshot sosit înaintea primului mesaj al noii conexiuni | **Defect B**, cauza principală a pierderii |

**Defect A.** Cât era neancorat, rezolverul arunca mesajele. Un snapshot nu se mai putea lega de un mesaj primit înainte
de el, deși un client real ține fluxul în buffer cât așteaptă REST-ul.

**Defect B.** La reconectare, snapshot-ul ajunge adesea înaintea primului update de pe conexiunea nouă. Lanțul părea
intact, snapshot-ul era amânat ca verificare, apoi noul update crea un gol, declarat real după 60 s. Book-ul rămânea
neancorat până la următorul snapshot, uneori ore întregi, deși ancora fusese deja primită. Am verificat pe primele 8
goluri din iulie 2026: în toate, rezolverul corectat ancorează la primul mesaj de după gol.

Ambele corecții respectă disponibilitatea: book-ul devine valid la momentul recepției, niciodată mai devreme (testat).

### Efectul corecțiilor

| | Vechi | Corectat |
|---|---|---|
| Valid din istoric | 26,2% | **40,7%** |
| Ancore / verificări | 5.942 / 2.259 | 7.168 / 3.310 |
| Goluri reale declarate | 527 | 35 |
| Ancore care nu se conectează | 103 | 1 |

| Lună | Valid vechi | Valid corectat |
|---|---|---|
| Iunie 2025 | 79,0% | 79,0% |
| Iulie 2025 – martie 2026 | 0% | 0% |
| Aprilie 2026 | 34,4% | 53,7% |
| Mai 2026 | 47,5% | 92,9% |
| Iunie 2026 | 51,1% | 99,6% |
| Iulie 2026 | 47,7% | 99,7% |
| August 2026 | 58,4% | 99,9% |
| Septembrie 2026 | 75,7% | 99,8% |
| Octombrie 2026 | 98,1% | 99,8% |

Aprilie e limitat de 168 de ore fără fișier. Noile ancore au trecut o verificare de conținut independentă: **3.388 de
comparații cu snapshot-uri, 0 nepotriviri**, cu o mediană de 992 de niveluri per comparație, într-o implementare de book
separată de kernelul de producție. Unde nimic nu s-a schimbat, codul nou reproduce vechile valori: 9.013.898 de valori
strict din 28–29 iunie 2025 sunt identice bit cu bit.

## 2. Experimentul

**Referința R:** replay strict din snapshot și actualizări continue. Checkpoint-urile sunt certificate doar pe cele mai
bune 500 de niveluri, fiindcă partea lor adâncă e replay-ul furnizorului. R nu e tratată ca adevăr în afara domeniului
certificat, iar comparațiile se fac doar acolo.

**Book-ul gol E(T):** pornit gol la T, nu vede niciodată un snapshot și primește aceleași mesaje. Am rulat fiecare E în
două feluri: *forțat* (emite tot, măsoară warm-up-ul de timp) și *certificat* (emite doar ce poate dovedi).

**Eșantion, declarat înainte de comparații** (`configs/d1_intervals.json`, sămânță 20261104): 176 de intervale și 688 de
starturi, pe 5 regimuri de colectare × 3 terțile de volatilitate × 3 terțile de lichiditate, plus intervale de stres și
starturi după goluri reale. Fiecare interval are startul la ancoră, trei momente interioare aleatoare și începutul celui
mai abrupt minut. Zilele sunt împărțite integral între calibrare și validare (202 zile).

**Eșantion mare, complementar.** Într-un lanț continuu, E(T) la momentul t e exact R(t) restrâns la nivelurile raportate
după T. O singură trecere prin referință dă deci răspunsul pentru orice start. Am aplicat asta pe tot timpul cu referință:
17,0 milioane de eșantioane de 1 s și 3,0 milioane de starturi. Identitatea a fost verificată contra book-urilor explicite
pe 357.880 de comparații: 3 diferențe de un rang, toate într-o singură rulare din mai 2026. Cauza e rotunjirea la milisecundă din
pasul pe vârste: ultimul update a sosit cu 832 µs înainte de T și a fost numărat ca primit după T. Book-urile sunt corecte.

Două modificări ale planului, ambele înainte de a citi comparații:
- Prima versiune avea o eroare (NaN în volatilitate), așa că am refăcut planul și am șters rulările necitite.
- Mostra „după gol” a căzut aproape toată în furtuna de reconectări din 24–27 mai 2026, așa că am adăugat 24 de intervale
  din perioade calme, după o regulă care depinde doar de lanț. Intervalele deja declarate au rămas neschimbate.

## 3. Starea curentă: convergență și persistență

**Teorie confirmată.** Pe 677 de rulări și 1.614.362 de comparații: 0 cantități greșite, 0 niveluri false, 0 încrucișări.
Book-ul gol doar lipsește niveluri, nu le greșește.

Rata la care un book pornit acum δ secunde nu are top-K exact (validare, 8,3 milioane de eșantioane):

| δ | Top 1 | Top 5 | Top 10 | Top 50 |
|---|---|---|---|---|
| 2 s | 0,05% | 69% | 79% | 92% |
| 10 s | 0,003% | 19% | 29% | 46% |
| 60 s | < 0,001% | 0,28% | 0,68% | 2,5% |
| 120 s | < 0,001% | 0,019% | 0,069% | 0,31% |
| 300 s | < 0,001% | < 0,001% | 0,002% | 0,007% |

Timpul după care top-K rămâne exact pe orizontul de o oră (validare, 1,48 milioane de starturi):

| | Mediană | p99 | p99,9 | Maxim | Recăderi |
|---|---|---|---|---|---|
| Top 1 | 0 s | 0 s | 2 s | 49 s | 0,1% |
| Top 5 | 4 s | 43 s | 120 s | 2.562 s | 7,5% |
| Top 10 | 5 s | 58 s | 518 s | 3.599 s | 9,5% |
| Top 50 | 9 s | 117 s | 1.677 s | 3.599 s | 10,5% |

Maximul atinge orizontul de o oră: un nivel de lângă vârf poate sta neatins mai mult de o oră. Am văzut unul de peste 4 ore
în top 50. Recăderile sunt reale: convergența nu e definitivă, fiindcă piața poate ajunge la un nivel adormit pe care book-ul
nu l-a văzut încă.

**Benzi de adâncime.** Partea de bandă care lipsește (validare, partea bid):

| Bandă | După 60 s | După 300 s | După 600 s | După 1.800 s | După 3.600 s |
|---|---|---|---|---|---|
| 1 bps: exact / deficit p99 | 98,1% / 0,04% | 99,996% / 0 | 100% / 0 | 100% / 0 | 100% / 0 |
| 5 bps | 80,5% / 15% | 99,2% / 0 | 99,9% / 0 | 99,999% / 0 | 100% / 0 |
| 10 bps | 38% / 32% | 89% / 2,2% | 97,2% / 0,1% | 99,95% / 0 | 99,998% / 0 |
| 25 bps | 2,3% / 58% | 25% / 20% | 50% / 7% | 86% / 0,5% | 97% / 0,02% |

Peste 25 bps referința devine rară (banda de 50 bps e comparabilă doar uneori), deci nu fac afirmații acolo.

**Features (validare, rulări forțate).** Spread-ul și `qi_1` coincid cu referința în 100% din cazuri după 2 s. `qi_top5` și
`qi_top10` ajung la 99,9% după 30–60 s, iar `qi_5bps` la 97,2% între 2 și 5 minute și la 99,9% între 5 și 10 minute. Tabelele pe
timp scurs sunt în `D1_TABLES.md` §6.

## 4. Features pe ferestre

Convergența stării nu validează fluxurile din trecut, așa că am măsurat separat, după `timp scurs − fereastră`.

- **Cu certificat:** o fereastră e emisă doar dacă vârful a fost certificat pe toată durata ei, deci valoarea e exactă de la
  început. Nu e nevoie de marjă suplimentară. Costul e că un singur moment necertificat sau vechi anulează fereastra.
- **Fără certificat (rulări forțate):** OFI coincide cu referința în 75–95% din cazuri în prima secundă după ce fereastra e
  plină, în 98,6–99,4% după 1–5 s, în 99,7–99,9% după 5–10 s și în 100% după 30 s. Ferestrele scurte și cele lungi se comportă
  la fel.
- **Adăugări/reduceri:** prima atingere a unui nivel are cantitatea anterioară necunoscută. În modul opțional ele apar
  doar dacă banda de 25 bps a fost certificată pe toată fereastra (disponibilitate 89% la 2 min, 76,5% la 15 min).
- **MLOFI pe ranguri:** rangul k e emis doar dacă a stat în domeniu pe toată fereastra.
- **Markouts și effective spread:** cer acoperire pe `w + h` plus vârsta maximă a cotației.

## 5. Politica de warm-up de timp

Regula a fost fixată înainte de rulare, iar valorile au ieșit doar din zilele de calibrare: rată agregată ≤ 0,1% și
percentila 99 a ratei orare ≤ 1% pentru top-K, respectiv deficit p99 ≤ 1% pentru benzi. Validarea a folosit politica
neschimbată.

| Familie | Warm-up | Validare: rată agregată | p99 orar | Cea mai proastă oră |
|---|---|---|---|---|
| Top 1 | 2 s | 0,052% | 0,33% | 1,0% |
| Top 5 | 120 s | 0,019% | 0,50% | 5,0% |
| Top 10 | 180 s | 0,018% | 0,03% | 11,4% |
| Top 50 | 300 s | 0,007% | 0,06% | 2,8% |
| Bandă 1 bps | 60 s | deficit p99 0,04% | | deficit maxim 74% |
| Bandă 5 bps | 300 s | deficit p99 0% | | deficit maxim 18% |
| Bandă 10 bps | 600 s | deficit p99 0,16% | | deficit maxim 17% |
| Bandă 25 bps | 1.800 s | deficit p99 0,54% | | deficit maxim 43% |

**Trece global, dar nu peste tot.** Pragul orar de 1% e depășit în validare de:
- top 5 și top 10 în regimul septembrie–octombrie 2026 (1,19% și 1,07%);
- top 5 și top 10 în terțila de volatilitate scăzută (1,71% și 1,43%);
- top 5, 10 și 50 în terțila de lichiditate mare (1,45%, 1,25%, 1,08%).

**Cazurile de eșec sunt orele calme și adânci, nu cele agitate.** Cea mai proastă oră (2026-08-09 00h, rv de 4,4 bps) are
11,4% nepotrivire la top 10. Când piața stă pe loc, nivelurile de lângă vârf nu se schimbă, deci book-ul gol nu le vede. În
ore volatile, top 10 se stabilizează în mediană în 1,5–2 s, față de 8–10 s în ore calme.

Pe rulările explicite de validare, politica a eșuat în 3 rulări, câte o singură secundă, cu 1–2 niveluri lipsă. Pe calibrare,
în niciuna.

**Articolul furnizorului.** „≤ 10 s pentru 500 de niveluri” nu se generalizează: după 10 s, top 50 diferă în 46% din
momente și top 5 în 19%. Regula „5 minute pentru tot ce e în 0,5%” nu ține la BTCUSDT nici pe 25 bps: după 300 s banda e
exactă doar 25% din timp, cu deficit p99 de 20%. Regula de 1 minut pentru top 5/10 dă 0,3–0,7% nepotrivire. Observația lor
de bază e însă confirmată: book-ul nu ține niveluri greșite.

## 6. Certificatul

Un book pornit gol e o submulțime a book-ului real: tot ce ține a fost raportat, deci e exact. Rezultă că `b ≤ b* < a* ≤ a`,
cu `b, a` cele mai bune niveluri proprii și `b*, a*` cele reale. Un bid ascuns poate sta doar strict între `b` și `a`. Dacă
toate tick-urile dintre ele au fost raportate pe partea bid, `b` e best bid-ul real, iar simetric pentru ask. La spread de un
tick nu e nimic între ele, deci vârful e exact prin protocol. De aici, frontiera certificată avansează doar peste tick-uri
contigue cunoscute.

| | Calibrare | Validare |
|---|---|---|
| Comparații | 848.085 | 766.277 |
| Vârf certificat | 99,96% | 99,95% |
| Vârf certificat greșit | **0** | **0** |
| Perechi de valori emise, toate features | 145.456.493 | 134.130.012 |
| Diferențe reale | **0** | **0** |
| Diferențe numerice pe `std_basis_*` | 13.417 (≤ 6e−8) | 2.267 (≤ 2e−8) |

Prima variantă nu avea zero: testul a scos la iveală valori pe ferestre parțiale, pe care le-am închis în modul reconstruit
(minime pe bandă, markouts, primul pas după reluarea certificării, baza după rebase). Ultimele corecții au venit după ce
văzusem validarea, așa că rezultatul „0 diferențe” nu e o validare independentă pentru acele margini de fereastră. Pentru
vârf și pentru cantități, 0 a fost la fel înainte și după.

Certificatul se adaptează singur: dă rezultat mai târziu când piața e calmă, în loc să greșească. Cât de repede devine
disponibil (cotă din momentele în care referința are valoarea, ambele seturi):

| Feature | 1–5 s | 10–30 s | 60–120 s | 300–600 s | 900–1.800 s | ≥ 1.800 s |
|---|---|---|---|---|---|---|
| Spread, `qi_1` | 99,9% | 99,97% | 99,98% | 100% | 100% | 100% |
| Adâncime top 10 | 25% | 86% | 99,6% | 99,97% | 100% | 100% |
| Bandă 5 bps | 4% | 42% | 87% | 99,1% | 100% | 100% |
| Bandă 10 bps | 1% | 17% | 58% | 94% | 99,9% | 100% |
| Bandă 25 bps | 0% | 1,6% | 13% | 46% | 82% | 95–97% |
| Bandă 50 bps | 0% | 0,5% | 3% | 15% | 41% | 70–82% |

**Excepția de la premisă.** Certificatul presupune că book-ul fluxului nu se încrucișează. Scanarea exhaustivă a găsit un
singur episod pe 491 de zile și 1.052.562.743 de mesaje: **2025-08-29, 06:37:36–06:55:52**, cu 14.565 de mesaje încrucișate
(0,0014%) și diferență de până la 292,5 USDT. Nivelul implicat (`ask 110987,4`) e raportat activ de flux, cu cantități care
se schimbă, deci nu e un update pierdut: un book clasic, pornit din snapshot, l-ar fi avut și el. **Cauza nu e stabilită**, iar
documentația furnizorului nu menționează episodul pentru Binance Futures. Crash-ul din octombrie 2025 nu are nicio
încrucișare.

La prima încrucișare certificatul se anulează pentru ambele niveluri. Nimic nu se emite timp de 600 s după ultimul mesaj încrucișat,
apoi book-ul repornește gol, deci certificarea revine doar din raportări făcute după acel moment, nu din expirarea carantinei.
Verificat pe episodul real: 0 rânduri certificate între prima încrucișare și 07:02:24. Valoarea de 600 s vine dintr-un singur
episod, deci e o măsură de prudență, nu un parametru validat. În cele 31 de secunde dinaintea primei încrucișări (06:37:05–06:37:36),
nivelul problematic apare deja ca vârf certificat, iar asta nu se poate detecta cauzal.

## 7. Limitele generalizării

Referința acoperă doar o parte din condiții:

| Regim | Referință | Fără referință |
|---|---|---|
| 102 ms | 2,4 zile (28–30 iunie 2025) | iulie 2025 |
| 52 ms | 20 de zile, la capătul regimului (11–30 aprilie 2026) | august 2025 – 10 aprilie 2026 |
| 26–28 ms | mai–octombrie 2026 | nimic |

Ce sprijină extinderea la lunile fără referință:

- **Vârstele nivelurilor, fără referință.** Un book pornit gol vede vârsta nivelurilor proprii. Pe lunile cu referință,
  estimarea asta se potrivește cu cea exactă (top 5 la 10 s: 21,2% vs. 20,2% în mai, 26,2% vs. 28,9% în august). Pe lunile
  fără referință, ratele la 10 s sunt în același interval (top 5: 7–28%), iar la warm-up-ul politicii rămân ≤ 0,08% pentru top
  5/10/50 și ≤ 0,25% pentru top 1. Estimarea folosește o zi din cinci și momente la cel puțin o oră de la repornire, deci **exclude tocmai perioadele cu goluri dese**.
- **Fluxul.** În regimul de 52 ms, cadența e aceeași (52 ms, ~19 mesaje/s), iar nivelurile per mesaj sunt 54–97, față de 68
  în aprilie 2026.
- **Integritatea.** Încrucișările sunt singurul semnal direct disponibil, iar scanarea lor e exhaustivă (§6).

Ce **nu** e acoperit:

- **Piața a fost mai activă fără referință.** Volatilitate orară mediană 34,5 vs. 30,0 bps, p99 138 vs. 115, notional median
  407 vs. 293 milioane USDT.
- **Extremele nu au referință.** Ora cea mai volatilă din istoric (2025-10-10 21h, 738 bps) e fără referință. Maximul cu
  referință e 396 bps.
- **În crash, acoperirea scade tocmai când contează.** Pe 2025-10-10 sunt 405 goluri reale și vârf certificat 88% din timp.
- **Exactitatea din cele 9,3 luni nu e măsurată direct,** ci dedusă din protocol, cu verificările de mai sus.

## 8. Goluri și durate utilizabile

După un gol real, book-ul se reconstruiește din gol și warm-up-ul reîncepe. Regula de 60 s rămâne cea strictă, dar o
cascadă de goluri costă acum o singură așteptare. Snapshot-urile ulterioare nu influențează când declarăm o valoare validă.

Perioada fără ancore are 1.866 de segmente vii și 6.581 de ore. Mediana duratei e mică în multe luni, dar segmentele scurte
sunt puține ca timp: **98,5% din timpul viu e în segmente de cel puțin o oră**.

| Lună | Segmente | Pe zi | Mediană (min) | p90 (min) | Sub 5 min | Viu | Viu după 5 min | Viu după 30 min |
|---|---|---|---|---|---|---|---|---|
| Iul 2025 | 73 | 2,4 | 179 | 1.506 | 22% | 99,9% | 99,2% | 96,5% |
| Aug 2025 | 105 | 3,4 | 5,4 | 1.281 | 50% | 99,8% | 99,2% | 96,6% |
| Sep 2025 | 134 | 4,5 | 29 | 1.025 | 37% | 99,8% | 98,7% | 94,4% |
| Oct 2025 | 568 | 18,3 | 0,16 | 126 | 79% | 99,6% | 97,9% | 92,3% |
| Nov 2025 | 176 | 5,9 | 5,0 | 834 | 50% | 99,7% | 98,5% | 93,9% |
| Dec 2025 | 149 | 4,8 | 0,42 | 1.111 | 60% | 99,8% | 99,1% | 96,1% |
| Ian 2026 | 172 | 5,5 | 1,0 | 884 | 59% | 99,8% | 98,9% | 95,3% |
| Feb 2026 | 148 | 5,3 | 0,06 | 524 | 63% | 99,9% | 99,1% | 95,8% |
| Mar 2026 | 334 | 10,8 | 110 | 285 | 20% | 94,9% | 91,8% | 78,0% |

Martie 2026 are 33 de ore fără fișier. Separat, 24–27 mai 2026 e o furtună de reconectări (850–2.850 de ancore pe zi).

## 9. Recomandare

### Ce recuperăm și în ce condiții

Acoperire măsurată în `btcusdt_v2` pe perioada fără ancore, bare de 1 minut (include cele 201 ore fără fișier):

| Familie | Disponibil | Calitate | Condiție |
|---|---|---|---|
| Spread, `qi_1` | 96,5% | exact | vârf certificat |
| Basis mid–index | 96,3% | exact | vârf certificat, mark proaspăt |
| Adâncime top 10, cost 1 BTC | 96,5% | exact | nivelurile în domeniu, de obicei 1–2 min |
| Benzi 1/5/10 bps | 96,5 / 96,4 / 96,2% | exact | frontiera acoperă banda, 2–15 min |
| Bandă 25 bps | 94,3% | exact | de obicei 30–60 min |
| Bandă 50 bps | 86,9% | exact | peste o oră |
| OFI pe 1 s / 30 s / 2 min / 15 min | 96,4 / 95,9 / 94,6 / 86,8% | exact | vârf certificat toată fereastra |
| MLOFI rang 5, 30 s | 95,9% | exact | rangul în domeniu toată fereastra |
| TW adâncime 5 bps 2 min / 25 bps 15 min | 94,5 / 82,5% | exact | banda în domeniu toată fereastra |
| Adăugări/reduceri 2 min / 15 min | 89,6 / 79,7% | exact | banda de 25 bps certificată toată fereastra |
| Effective spread 2 min, markout 30 s pe 15 min | 94,6 / 85,4% | exact | fereastră + `h` + vârsta cotației |

„Exact” înseamnă: 0 diferențe față de referință unde există, iar în lunile fără referință, exact dacă premisele protocolului
țin. Ferestrele lungi pierd mai mult pentru că regula strictă cere acoperire de 100%, deci o singură secundă veche anulează
15 minute. Asta ține de D4, nu de reconstrucție.

**Nu recuperăm:** adâncimea dincolo de certificat, pe care niciun warm-up nu o poate garanta, și nici intervalele de gol.

### Sporul măsurat

`btcusdt_v2`, bare de 1 minut. Tabelele pentru toate seturile de bare și familiile sunt în `docs/DATASET_V2.md`.

| Familie | Strict vechi | Tot istoricul | Fără ancore | Crash, 9–11 oct. 2025 |
|---|---|---|---|---|
| Vârf de book | 26,2% | **97,3%** | 96,5% | 96,1% |
| Adâncime top 10, bandă 5 bps | 26,2% | **97,3%** | 96,4–96,5% | 96,0–96,1% |
| Bandă 25 bps | 26,2% | 94,9% | 94,3% | 91,6% |
| OFI 2 min | 26,2% | 94,4% | 94,6% | 91,4% |
| OFI 15 min | 26,2% | 86,4% | 86,8% | 79,6% |

Estimarea anterioară, din 27 de zile, era bună pentru vârf (96,5%) și optimistă pentru ferestre (OFI 15 min: 90,5%).

**Acoperirea e mai mică pe bare bazate pe evenimente, pentru că ele se adună când fluxul cade.** Vârf de book măsurat: `time_1m` 97,3%, `volume_50btc`
95,8%, `dollar_5musdt` 96,1%, `cusum_5bps` 92,4%, `cusum_adapt_c1` 90,7%. Pe 9–11 octombrie 2025: 96,1%, 79,0%, 79,8%, 24,7% și 19,6%. Absența din crash nu e aleatoare,
deci un model antrenat pe aceste bare nu vede tocmai cele mai agitate momente.

### Ce rămâne incert

1. Exactitatea în cele 9,3 luni e dedusă, nu măsurată (§7).
2. Cauza episodului din 2025-08-29 nu e cunoscută.
3. Comportamentul la volatilitate peste 396 bps pe oră nu are referință, iar în crash acoperirea pe bare e mică.
4. Fișierele noastre pot deveni depășite: furnizorul anunță că adaugă retroactiv snapshot-uri de oră în arhivă. Nu am verificat
   dacă a ajuns la această perioadă (nu am folosit cheia API). Dacă da, ar înlocui mare parte din reconstrucție cu ancore reale.

## 10. Modul opțional

Implementat separat, ca `book_mode: rebuilt`. Modul strict rămâne implicit, iar rezolverul vechi e disponibil cu
`legacy_chain: true`, pentru reproducerea dataseturilor existente.

- `book_state` 0: ancorat. 9: fără ancoră, vârf certificat. 10: fără ancoră, vârf necertificat, features `null`. Nivelurile
  rămân separate în date, iar `cert_bid_floor_dist_bps` și `cert_ask_ceiling_dist_bps` arată până unde ajunge certificatul.
- Un snapshot care se conectează readuce book-ul în nivelul ancorat. După un gol real, book-ul repornește gol.
- Fără warm-up de timp: nu emite nimic ce nu poate dovedi.

**Demonstrație** (`configs/rebuilt_demo.yaml`, `scripts/run_rebuilt_demo.sh`), două subseturi fără ancore, cu dataseturi separate.
Datasetul strict are 0% pe aceste zile.

| Subset | Bare | Vârf certificat | Bandă 5 bps | Bandă 25 bps | OFI 2 min | OFI 15 min |
|---|---|---|---|---|---|---|
| 14–16 dec. 2025 | `time_1m` | 99,7% | 99,7% | 98,2% | 98,5% | 93,4% |
| | `volume_50btc` | 99,2% | 99,2% | 97,1% | 94,2% | 81,0% |
| | `cusum_5bps` | 99,2% | 99,2% | 96,8% | 94,1% | 80,8% |
| 9–11 oct. 2025 | `time_1m` | 96,1% | 95,9% | 91,2% | 91,4% | 79,2% |
| | `volume_50btc` | 79,0% | 79,0% | 72,6% | 68,3% | 54,0% |
| | `cusum_5bps` | 24,7% | 24,7% | 22,1% | 19,5% | 14,6% |

Verificarea independentă a celor două dataseturi (`scripts/verify_dataset.py`, pe 2025-12-15 și 2025-10-10): 60.295 de bare,
4.654 de recalculări brute-force, 0 erori de conservare, 0 valori sub măști invalide, 0 coloane neînregistrate. Demonstrația a
fost rulată înainte de carantină. Aceasta nu o afectează, fiindcă pe acele zile nu există încrucișări.

## 11. Decizii pentru tine

1. **Regenerez strict cu rezolverul corectat?** Recomand da: 26,2% → 40,7%, cu verificare de conținut curată.
2. **Generez un dataset separat în modul opțional pe tot istoricul?** Recomand da, sub alt nume, cu `book_state` păstrat ca
   feature, ca modelele să poată separa nivelurile.
3. **Tratezi barele din crash separat?** Pentru CUSUM, acoperirea din crash e prea mică pentru a fi ignorată. Recomand să
   raportăm acoperirea pe bare la fiecare experiment.
4. **Contactăm furnizorul?** Recomand să întrebăm despre stadiul backfill-ului pentru iulie 2025 – aprilie 2026 și despre
   2025-08-29 06:37–06:56. Aștept acordul tău.
5. **Corectez bug-ul de markout din strict?** Vezi mai jos. Recomand da, odată cu regenerarea.

## 12. Constatare separată de D1

Bug în kernel, corectat între timp în ambele moduri (vezi `docs/REGEN_READINESS.md`). La sfârșitul fiecărei ore, scadențele de markout
de după ultimul eveniment erau procesate pe starea de sfârșit de oră. Ultimele `h` secunde de tranzacții erau pierdute sau evaluate
pe un mid vechi, iar `mo_t` rămânea nesortat la graniță. Nu introducea date din viitor.

## 13. Reproducere

```bash
P=.venv/bin/python
$P scripts/audit_snapshot_grouping.py 6     # părți de snapshot, conexiuni
$P scripts/d1_chain_pass.py 4               # efectul corecțiilor, goluri (~70 min)
$P scripts/d1_hour_table.py
$P scripts/d1_select_intervals.py           # planul, înainte de orice comparație
$P scripts/d1_experiment.py --workers 6     # R vs. E
$P scripts/d1_age_pass.py 5                 # eșantion mare + verificare de conținut
$P scripts/d1_age_pass.py 4 0 empty         # vârste fără referință
$P scripts/d1_no_reference.py 3             # modul opțional pe zile eșantion
$P scripts/d1_cross_scan.py 6               # încrucișări, exhaustiv
$P scripts/d1_analyze.py age --split calibration
$P scripts/d1_analyze.py runs --split calibration
$P scripts/d1_analyze.py policy             # fixează politica
$P scripts/d1_analyze.py age --split validation
$P scripts/d1_analyze.py runs --split validation
$P scripts/d1_analyze.py validate
$P scripts/d1_analyze.py validate_runs --split validation
$P scripts/d1_analyze.py validate_runs --split calibration
$P scripts/d1_coverage.py; $P scripts/d1_generalization.py; $P scripts/d1_failures.py
$P scripts/d1_identity_check.py 16 6; $P scripts/d1_summarize.py; $P scripts/d1_demo_report.py
$P scripts/d1_report_tables.py              # docs/D1_TABLES.md
$P scripts/d1_strict_regression.py          # strict bit cu bit pe 2025-06-28/29
bash scripts/run_rebuilt_demo.sh            # demonstrația
```

Articolul furnizorului e arhivat în `docs/vendor_docs_2026-11-04/`.
