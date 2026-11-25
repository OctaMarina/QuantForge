# This is the research pipeline for QuantForge

1. Select an asset and build multiple datasets with multiple bars and multiple frequencies in order to select the best 
configuration of this specific asset.
2. **Evaluation of Sampling Methods:** Various sampling techniques (e.g., CUSUM with different thresholds) will be tested on one or more of the previously evaluated datasets, aiming to obtain the most favorable statistical properties.
3. **Primary Strategy:** Consider ca sunt doua variante, pornesc de la o varianta economica si aleg primary strategy, sau pornesc
de la date pe o perioada mai mica din dataset si aleg un primary strategy ce il testez pe o perioada nevazuta

"folosește trend scanning ca să judeci și să construiești strategia primară, și triple barrier ca etichetă pentru meta-model, cu barierele identice cu ieșirea reală."