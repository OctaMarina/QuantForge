# Statut final pe familii

Data verificării: **2026-10-20**. Mediul de build are acces la PyPI și GitHub, dar **toate domeniile furnizorilor de date
sunt blocate de proxy (HTTP 403 la CONNECT)**: data.binance.vision, api/fapi.binance.com, api.exchange.coinbase.com,
www.deribit.com, api.alternative.me, farside.co.uk, web.archive.org, datasets.tardis.dev, api.bybit.com, precum și
WebSocket-urile Binance/Bybit/Deribit (verificat cu `dry-run`, `download` și `live --seconds 6`). Documentația a fost
citită prin GitHub (repo-urile oficiale Binance și Bybit) și prin paginile publice de documentație. De aceea,
„istoric verificat” înseamnă aici exclusiv ce s-a putut confirma din documentație și din eșantioanele reale disponibile;
nicio acoperire istorică nu a fost confirmată prin răspunsuri reale ale API-urilor.

Legendă statut (secțiunea 1 a cerinței):
**IV** = implementat și verificat · **IN** = implementat, dar netestat pe date reale · **ID** = indisponibil din sursa curentă
(istoric) · **NI** = neimplementat.

| Familie | Sursă aleasă | Istoric verificat | Live | Exactitate metodologică | Statut temporal | Motiv tipic pentru null |
|---|---|---|---|---|---|---|
| Bare time / volume / quote-notional | Binance `trades` (execuții individuale), arhiva data.binance.vision | Format și reguli verificate în documentație; **date reale: doar fixture-ul ETHUSDT 2020-08-14 10:00–15:00** (69 806 rânduri, 70 duplicate exacte eliminate după `trade_id`, ID-uri contigue) | colector `@trade` + replay (testat cu loguri simulate) | **IV**: aritmetică întreagă exactă, prag atins sau depășit, tranzacția integrală inclusă, stare la restart, invarianță batch/restart bit cu bit | market-time (nivel 1); observed-availability din loguri de recepție (nivel 2) | `feed_gap` (interval incomplet), segment nou după gol nerecuperabil |
| Randamente (calendar + bare) | trades | ca mai sus | da | **IV** | închiderea barei | `insufficient_history`, `stale_input`, `feed_gap` |
| Volatilitate realizată (grilă 1 min + bare) | trades | ca mai sus | da | **IV** (estimator RV pe grilă explicită) | închiderea barei | `insufficient_history`, `stale_input` |
| Durată / intensitate | trades | ca mai sus | da | **IV** | închiderea barei | `zero_denominator` (bară instantanee), `feed_gap` |
| Flux agresor observat (isBuyerMaker) | trades | mapare verificată în documentația Binance | da | **IV** (buy+sell=total testat) | închiderea barei | `incomplete_coverage` |
| Tick rule | trades | — | da | **IV** (inițializare fără completare retroactivă) | închiderea barei | `incomplete_coverage` (înainte de prima schimbare de preț) |
| Kyle λ / Hasbrouck λ (tranzacții în bară; rolling pe bare) | trades | — | da | **IV** față de `statsmodels.OLS` (coef. și t); textul AFML nu a putut fi verificat | închiderea barei | `insufficient_history`, `invalid_regression` |
| SADF / QADF / CADF (windowed) | bare | — | da | **IV** față de referință brute-force OLS (grilă completă de începuturi); fidelitate față de textul AFML neverificată (vezi FORMULAS.md) | endpoint t | `insufficient_history`, `feed_gap` |
| SDFC (Chow-type DF) | bare | — | da | **IV** față de referință; atribuit doar endpointului | endpoint t | idem |
| CUSUM Chu–Stinchcombe–White | bare | — | da | **IV** față de referință naivă | endpoint t | idem |
| CUSUM Brown–Durbin–Evans | bare | — | da | **IV**: egal cu `rcusum` din `statsmodels.recursive_olsresiduals` pe fereastră | endpoint t | idem |
| Premium USD vs USDT (brut, ajustat FX, zilnic, z90) | Coinbase `BTC-USD`/`ETH-USD`/`USDT-USD` 1m + Binance klines 1m | doc. Coinbase verificată (bucket start, max 300/req, minute lipsă fără tranzacții); acoperire istorică **neverificată** (blocat) | (poate fi reconstruit din arhive) | **IN** (testat pe fixture-uri manuale, unități explicite) | sfârșitul minutului / zilei | `missing_source_data`, `source_unreachable`, `stale_input` |
| Trend 30 zile scalat la volatilitate | Binance 1d klines | neverificat (blocat) | — | **IN** | sfârșitul zilei | `insufficient_history`, `zero_denominator` |
| Basis perp–spot (last; mark/index), Premium Index, funding | Binance USD-M klines/markPrice/indexPrice/premiumIndex + `/fapi/v1/fundingRate` | endpoint-uri verificate în conectorul oficial; istoric neverificat | — | **IN**; Premium Index preluat direct; interval funding observat, nu presupus 8h | sfârșitul minutului / momentul funding | `missing_source_data`, `stale_input` |
| Fluxuri ETF spot BTC și ETH (Farside) | Farside + vintage-uri din poll-uri live sau capturi Wayback | structura paginii verificată (US$m, paranteze, rânduri sumar); **momentul primei publicări NU e dovedit de tabelul curent** | poll la 15 min, cu `received_at` | **IN** (parser testat pe o pagină sintetică fidelă structurii verificate) | strict: doar din dovadă (recepție / captură); research: lag presupus, fișier separat | `unknown_publication_time` (istoric fără dovadă), `stale_input` |
| Fear & Greed | Alternative.me + vintage-uri | API verificat (timestamp = ziua economică, fără oră de publicare) | poll la 15 min | **IN** | ca la ETF | `unknown_publication_time` |
| DVOL BTC/ETH | Deribit `get_volatility_index_data` | parametri verificați; unitate nedeclarată în API (articolul de metodologie Deribit ⇒ puncte procentuale anualizate); adâncime istorică neverificată | canal `deribit_volatility_index.*` | **IN** | sfârșitul lumânării (conservator) | `missing_source_data`, `stale_input` |
| Skew 25Δ la 30 zile | Deribit tickers (live) / Tardis `options_chain` (doar 1 a lunii gratuit) | **ID**: nu există lanț istoric gratuit și continuu | abonare `ticker.*.100ms` la opțiunile 7–60 zile | **IN** (interpolare în delta + varianță totală, fără extrapolare; testat manual) | snapshot (quote-uri strict anterioare) | `not_available_in_period`, `insufficient_book_depth` |
| Lichidări | Binance `forceOrder` (**eșantionat**: ultima lichidare per 1000 ms) + Bybit `allLiquidation` (toate, preț de faliment) | **ID** (fără istoric gratuit) | da, cu loguri de conexiune | **IN** (mapări de semn testate: Binance S = latura ordinului, Bybit S = latura poziției) | timp de eveniment / recepție | `not_available_in_period`, `feed_gap` (deconectare ≠ zero) |
| L2: adâncime, imbalance, retragere/refacere, concentrări, persistență, interacțiuni | Binance diff depth @100ms + snapshot REST (live) / Tardis `incremental_book_L2` | **ID** (fără istoric gratuit continuu) | da | reconstrucție **IV** pe fixture-uri reale (snapshot+diff Binance USD-M → golul e detectat și refuzat; Tardis Deribit inverse) și sintetice (benzi incrementale = recalcul brute-force) | mesaje de carnet strict anterioare deciziei | `not_available_in_period`, `insufficient_book_depth`, `stale_input` |
| Cost estimat din carnet | L2 | **ID** | da | **IV** pe fixture manual (VWAP, fără dublarea spread-ului, adâncime insuficientă → null) | decizie | `insufficient_book_depth` |
| Spread din quote-uri (bookTicker) | bookTicker live / arhiva USD-M bookTicker / Tardis quotes | spot: fără istoric gratuit | da | **IV** pe fixture real Binance BTCUSDT (2021-01-08) | decizie / interval bară | `not_available_in_period`, `invalid_quote`, `stale_input` |
| Bare USD-notional | — | — | — | **NI** (necesită conversie USD per tranzacție, contemporană și cauzală) | — | opțiunea e refuzată de validare |
| Fallback time bars din klines native | — | — | — | **NI** (opțional în cerință) | — | refuzat de validare |
| Variante HAC pentru λ, frontiere critice CSW | — | — | — | **NI** (doar erori standard clasice; nicio substituție tăcută) | — | — |

## Ce a fost efectiv rulat în acest mediu

* `pytest`: **80 de teste trecute** (vezi `TEST_RESULTS.md`), inclusiv suita de absență a lookahead-ului pe date reale.
* `sample-real`: dataset real mic (ETHUSDT, 5 ore) în `sample_output/` cu manifest, registre, calitate și acoperire.
* `dry-run`, `download`, `build`, `live --seconds 6`: rulate; rezultat = toate sursele blocate, raportate explicit
  (`source_unreachable`), fără date fabricate.
* Fallback-ul fără numba (`NUMBA_DISABLE_JIT=1`): testele de bare/flux/structurale trec; sumele float ale barelor sunt
  identice bit cu bit pe eșantionul verificat.

## Ce trebuie rulat pe o mașină cu acces la internet

1. `python crypto_dataset.py dry-run` – confirmă accesul și volumele (listarea S3 a arhivei Binance).
2. `python crypto_dataset.py download` apoi `build` – primul build real; manifestul raportează și verificarea
   încrucișată trades→1m vs klines native (`kline_crosscheck_1m`).
3. `python crypto_dataset.py live` lăsat să ruleze – singura cale gratuită pentru L2, lichidări, skew și pentru
   dovada point-in-time a ETF/Fear & Greed; apoi `replay-live`.
