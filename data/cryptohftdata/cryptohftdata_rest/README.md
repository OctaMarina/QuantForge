# Descărcare CryptoHFTData cu token reînnoit automat

Pentru Linux, Python 3.10+ și mediul virtual existent `.cryptohftdata`.
Acest downloader folosește REST cu API key-ul obișnuit din cont. Nu folosește cheile S3 temporare.

## 1. Instalare

Oprește vechiul downloader cu Ctrl+C dacă încă rulează. Nu rula cele două scripturi simultan în același folder de date.
Dezarhivează conținutul acestui pachet în:

```bash
cd ~/Desktop/QuantForge/data/cryptohftdata
```

Fișierele `download_cryptohftdata_rest.py` și `requirements_rest.txt` trebuie să fie direct în acest director, lângă downloaderul S3 existent. Nu muta și nu șterge folderul `cryptohftdata_raw`.

Dacă promptul arată deja `(.cryptohftdata)`, mediul virtual este activ. Altfel activează mediul existent din locația unde l-ai creat. Dacă folderul lui este chiar aici, comanda este `source .cryptohftdata/bin/activate`.

```bash
python --version
python -m pip install -r requirements_rest.txt
df -h .
```

## 2. Configurare `.env`

Editează fișierul existent:

```bash
nano .env
```

Adaugă o singură linie, înlocuind exemplul cu API key-ul obișnuit din dashboard:

```dotenv
CRYPTOHFTDATA_API_KEY='cheia_ta_API_completa'
```

Folosește API key-ul, nu `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_SESSION_TOKEN` sau un JWT. Ghilimelele trebuie închise. Liniile AWS existente pot rămâne; acest script nu le utilizează. Nu trimite nimănui conținutul `.env`.

În nano: Ctrl+O, Enter, Ctrl+X.
Scriptul citește singur `.env`; nu trebuie să folosești `export` sau `dotenv run`.

## 3. Pornește verificarea și continuarea descărcării

Din același director, cu mediul virtual activ:

```bash
python -u download_cryptohftdata_rest.py \
  --start 2025-06-28 \
  --end 2026-10-31 \
  --exchanges binance_futures \
  --symbols BTCUSDT \
  --types orderbook trades ticker liquidations open_interest mark_price \
  --output ./cryptohftdata_raw
```

Intervalul este inclusiv, în UTC. Sunt șase tipuri; pentru Binance Futures informația de funding este în datele `mark_price`.

Începem din nou de la 2025-06-28 pentru a verifica întregul interval cerut, inclusiv ce există deja. Această dată este limita aleasă pentru această rulare, nu o afirmație că fiecare tip are date pentru fiecare oră începând de atunci.

Scriptul:

- Verifică API key-ul la pornire și solicită un JWT. Îl reînnoiește înaintea expirării; la un refuz 401/403 reîncearcă o dată cu token nou. Un refuz persistent oprește rularea, fără a șterge datele existente.
- Parcurge fiecare zi, oră și tip cerut. Reutilizează căile existente `binance_futures/YYYY-MM-DD/HH/BTCUSDT_tip.parquet` și `.parquet.zst`.
- La prima întâlnire a unui fișier local, citește toate coloanele Parquet pentru a verifica dacă se pot decoda. Pentru arhivele cu compresie Zstandard exterioară decomprimă temporar un singur fișier orar pe disc; ai nevoie de spațiu liber pentru acesta și pentru datele noi.
- Păstrează în SQLite rezultatul verificării. La reluare, fișierele cu aceeași dimensiune și dată de modificare sunt sărite rapid. `--revalidate` forțează o nouă citire completă.
- Descarcă un fișier absent sau invalid într-un fișier temporar `.rest.part`; îl promovează la numele final numai după verificarea transferului și citirea Parquet. La întrerupere, reia întregul fișier orar incomplet, nu întregul istoric.
- Reîncearcă erorile temporare și respectă `Retry-After`. Înregistrează separat răspunsurile 404.

Prima trecere prin cei aproximativ 33 GB existenți poate dura. `VALIDATE local ...` înseamnă citire locală, nu redescărcare. `DOWNLOADED ...` înseamnă fișier nou descărcat. Lasă terminalul deschis și evită suspendarea calculatorului cât rulează.

## 4. Reluare și controlul rezultatului

Pentru oprire normală: Ctrl+C. Pentru reluare: aceeași comandă, același folder de output. Nu șterge baza `rest_download_state.sqlite` dacă vrei să păstrezi verificările înregistrate.

La final sau la oprire normală se exportă:

- `cryptohftdata_raw/rest_manifest.csv`: starea fiecărui fișier procesat, dimensiunea și numărul de rânduri pentru cele valide.
- `cryptohftdata_raw/rest_unavailable_or_errors.csv`: obiectele procesate cu 404 sau erori. Fiecare linie include calea cu data, ora și tipul.

Poți vedea începutul raportului cu:

```bash
head -n 20 cryptohftdata_raw/rest_unavailable_or_errors.csv
```

Un raport care are doar antetul nu conține probleme înregistrate, dar trebuie și ca rularea să fi ajuns la capătul intervalului. Dacă ai întrerupt procesul, orele neparcurse încă nu sunt în raport. Fișierele 404 sunt reîncercate la următoarea rulare.

Verificarea Parquet detectează multe forme de corupere, dar NU compară fișierul local cu un checksum de la furnizor, nu detectează orice modificare silențioasă și nu demonstrează că furnizorul a capturat toate evenimentele pieței. Un 404 nu înseamnă automat zero lichidări. Pentru analiza finală sunt necesare și controale pe timestampuri, continuitate și lacunele declarate de furnizor.

Tokenul se reînnoiește automat cât timp API key-ul și accesul contului rămân valide și serviciul este disponibil. Nu există o garanție că o întrerupere de rețea, revocare de cheie sau eroare a serviciului nu va opri procesul. Protecția principală este reluarea fără pierderea fișierelor validate.

## Validare și documentație

Sunt incluse nouă teste locale: reînnoire token, refuz de autentificare, rate limit, erori de server, 404, transfer întrerupt, lungime incorectă, arhive Zstandard și reluarea/verificarea fișierelor locale. Pentru rulare:

```bash
python -m unittest test_downloader.py -v
```

Fluxul autentificat cu contul tău nu a putut fi testat fără API key-ul tău; acesta rămâne doar la tine.

Documentație oficială:
- https://www.cryptohftdata.com/docs/rest-authentication
- https://www.cryptohftdata.com/docs/rest-trades
- https://www.cryptohftdata.com/docs/rest-mark-price
- https://www.cryptohftdata.com/blog/flat-file-access-now-live
