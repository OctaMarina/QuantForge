# Audit of the CryptoHFTData BTCUSDT (Binance USD-M) history 2025-06-28 → 2026-10-31


Generated 2026-11-03 16:38 UTC by `chdds audit-report`. Raw files were read only; no raw file was modified. Tables are also written as CSV under `derived/audit/`.


## 1. Inventory and manifests (EXHAUSTIVE over the 11,784-hour grid × 6 streams)

Manifest/file-system cross-checks:

```
{
 "manifest_keys": 70704,
 "manifest_unique_keys": 70704,
 "manifest_status": {
  "unavailable": 350,
  "ok": 70354
 },
 "unavailable_file_rows": 350,
 "unavailable_equals_manifest_subset": true,
 "grid_hours": 11784,
 "grid_keys_missing_from_manifest": 0,
 "ok_but_missing_on_disk": 0,
 "on_disk_but_not_ok": 0,
 "size_mismatch_vs_manifest": 0,
 "ok_with_zero_rows": 0,
 "s3_log_rows": 12146,
 "s3_log_status": {
  "ok": 12140,
  "error": 5,
  "list_error": 1
 },
 "s3_ok_keys_not_ok_in_rest": 0,
 "fs_formats": {
  "zst": 59769,
  "parquet": 10585
 }
}
```

| stream | hours | files | GB |
|---|---|---|---|
| liquidations | 11,784 | 11,709 | 0.039 |
| mark_price | 11,784 | 11,755 | 1.129 |
| open_interest | 11,784 | 11,778 | 0.017 |
| orderbook | 11,784 | 11,578 | 210.973 |
| ticker | 11,784 | 11,755 | 1.542 |
| trades | 11,784 | 11,779 | 18.184 |


Unavailable intervals longer than one hour (end exclusive; full list incl. single hours in `derived/audit/unavailable_intervals.csv`):

| stream | start | end_excl | hours |
|---|---|---|---|
| trades | 2025-06-28 00:00 | 2025-06-28 05:00 | 5 |
| orderbook | 2025-06-28 00:00 | 2025-06-28 05:00 | 5 |
| orderbook | 2026-03-30 15:00 | 2026-04-01 10:00 | 43 |
| orderbook | 2026-04-02 16:00 | 2026-04-05 20:00 | 76 |
| orderbook | 2026-04-05 23:00 | 2026-04-08 08:00 | 57 |
| orderbook | 2026-04-08 14:00 | 2026-04-09 15:00 | 25 |
| ticker | 2025-06-28 00:00 | 2025-06-28 05:00 | 5 |
| ticker | 2026-04-23 17:00 | 2026-04-24 17:00 | 24 |
| liquidations | 2025-06-28 00:00 | 2025-06-28 05:00 | 5 |
| liquidations | 2026-04-23 18:00 | 2026-04-24 17:00 | 23 |
| liquidations | 2026-05-15 22:00 | 2026-05-16 00:00 | 2 |
| liquidations | 2026-05-16 21:00 | 2026-05-16 23:00 | 2 |
| liquidations | 2026-05-17 07:00 | 2026-05-17 09:00 | 2 |
| liquidations | 2026-05-30 20:00 | 2026-05-30 22:00 | 2 |
| liquidations | 2026-10-17 08:00 | 2026-10-17 10:00 | 2 |
| open_interest | 2025-06-28 00:00 | 2025-06-28 06:00 | 6 |
| mark_price | 2025-06-28 00:00 | 2025-06-28 05:00 | 5 |
| mark_price | 2026-04-23 17:00 | 2026-04-24 17:00 | 24 |


Single-hour liquidation files missing: 37. A missing liquidation file is NOT treated as zero liquidations (default policy `unknown`).


Hours with all six files present: 11,506 of 11,784 (97.641%). File presence is not completeness — see sections below.


## 2. Schemas, versions, writers and repair markers (EXHAUSTIVE)

### trades

| schema_fp | created_by | kv_metadata | first_day | last_day | files | columns |
|---|---|---|---|---|---|---|
| 7db46ddee7b2 | parquet-rs version 55.1.0 | {} | 2025-06-28 | 2026-09-25 | 10,901 | [received_time:int64, event_time:int64, symbol:string, trade_id:int64, price:string, quantity:string, trade_time:int64, is_buyer_maker:bool, order_type:string] |
| 7db46ddee7b2 | parquet-rs version 55.2.0 | {} | 2026-09-25 | 2026-09-28 | 79 | [received_time:int64, event_time:int64, symbol:string, trade_id:int64, price:string, quantity:string, trade_time:int64, is_buyer_maker:bool, order_type:string] |
| 7db46ddee7b2 | parquet-rs version 60.0.0 | {} | 2026-09-28 | 2026-10-31 | 799 | [received_time:int64, event_time:int64, symbol:string, trade_id:int64, price:string, quantity:string, trade_time:int64, is_buyer_maker:bool, order_type:string] |

Row count read vs parquet footer mismatches: 0.

### orderbook

| schema_fp | created_by | kv_metadata | first_day | last_day | files | columns |
|---|---|---|---|---|---|---|
| 90d8e6b50a7d | parquet-rs version 55.1.0 | {} | 2025-06-28 | 2025-06-30 | 54 | [received_time:int64, event_time:int64, transaction_time:int64, symbol:string, event_type:string, first_update_id:int64, final_update_id:int64, prev_final_update_id:int64, last_update_id:int64, side:string, price:string, quantity:string] |
| 4a9a44aa2958 | parquet-rs version 55.1.0 | {} | 2025-06-30 | 2026-09-25 | 10,645 | [received_time:int64, event_time:int64, transaction_time:int64, symbol:string, event_type:string, first_update_id:int64, final_update_id:int64, prev_final_update_id:int64, last_update_id:int64, side:string, price:string, quantity:string, order_count:int64] |
| 4a9a44aa2958 | parquet-cpp-arrow version 25.0.1 | {"chd.repairs": "[\"binance_duplicate_depth_snapshots/v1\"]"} | 2026-09-01 | 2026-09-01 | 1 | [received_time:int64, event_time:int64, transaction_time:int64, symbol:string, event_type:string, first_update_id:int64, final_update_id:int64, prev_final_update_id:int64, last_update_id:int64, side:string, price:string, quantity:string, order_count:int64] |
| 4a9a44aa2958 | parquet-rs version 55.2.0 | {} | 2026-09-25 | 2026-09-28 | 79 | [received_time:int64, event_time:int64, transaction_time:int64, symbol:string, event_type:string, first_update_id:int64, final_update_id:int64, prev_final_update_id:int64, last_update_id:int64, side:string, price:string, quantity:string, order_count:int64] |
| 4a9a44aa2958 | parquet-rs version 60.0.0 | {} | 2026-09-28 | 2026-10-31 | 799 | [received_time:int64, event_time:int64, transaction_time:int64, symbol:string, event_type:string, first_update_id:int64, final_update_id:int64, prev_final_update_id:int64, last_update_id:int64, side:string, price:string, quantity:string, order_count:int64] |

Row count read vs parquet footer mismatches: 0.

### ticker

| schema_fp | created_by | kv_metadata | first_day | last_day | files | columns |
|---|---|---|---|---|---|---|
| 719b2ee3782a | parquet-rs version 55.1.0 | {} | 2025-06-28 | 2026-09-25 | 10,877 | [received_time:int64, event_time:int64, symbol:string, price_change:string, price_change_percent:string, weighted_average_price:string, last_price:string, last_quantity:string, open_price:string, high_price:string, low_price:string, base_asset_volume:string, quote_asset_volume:string, statistics_ope |
| 719b2ee3782a | parquet-rs version 55.2.0 | {} | 2026-09-25 | 2026-09-28 | 79 | [received_time:int64, event_time:int64, symbol:string, price_change:string, price_change_percent:string, weighted_average_price:string, last_price:string, last_quantity:string, open_price:string, high_price:string, low_price:string, base_asset_volume:string, quote_asset_volume:string, statistics_ope |
| 719b2ee3782a | parquet-rs version 60.0.0 | {} | 2026-09-28 | 2026-10-31 | 799 | [received_time:int64, event_time:int64, symbol:string, price_change:string, price_change_percent:string, weighted_average_price:string, last_price:string, last_quantity:string, open_price:string, high_price:string, low_price:string, base_asset_volume:string, quote_asset_volume:string, statistics_ope |

Row count read vs parquet footer mismatches: 0.

### liquidations

| schema_fp | created_by | kv_metadata | first_day | last_day | files | columns |
|---|---|---|---|---|---|---|
| 221043cf69ab | parquet-rs version 55.1.0 | {} | 2025-06-28 | 2026-09-25 | 10,839 | [received_time:int64, event_time:int64, symbol:string, side:string, order_type:string, time_in_force:string, quantity:string, price:string, average_price:string, order_status:string, last_filled_quantity:string, filled_quantity:string, trade_time:int64] |
| 221043cf69ab | parquet-rs version 55.2.0 | {} | 2026-09-25 | 2026-09-28 | 79 | [received_time:int64, event_time:int64, symbol:string, side:string, order_type:string, time_in_force:string, quantity:string, price:string, average_price:string, order_status:string, last_filled_quantity:string, filled_quantity:string, trade_time:int64] |
| 221043cf69ab | parquet-rs version 60.0.0 | {} | 2026-09-28 | 2026-10-31 | 791 | [received_time:int64, event_time:int64, symbol:string, side:string, order_type:string, time_in_force:string, quantity:string, price:string, average_price:string, order_status:string, last_filled_quantity:string, filled_quantity:string, trade_time:int64] |

Row count read vs parquet footer mismatches: 0.

### open_interest

| schema_fp | created_by | kv_metadata | first_day | last_day | files | columns |
|---|---|---|---|---|---|---|
| f1c96e8c0c70 | parquet-rs version 55.1.0 | {} | 2025-06-28 | 2025-06-28 | 1 | [received_time:int64, symbol:string, sum_open_interest:string, sum_open_interest_value:string, timestamp:int64] |
| f1c96e8c0c70 | parquet-cpp-arrow version 25.0.1 | {"chd.repairs": "[\"binance_futures_oi_cross_hour_dedup/v1\"]"} | 2025-06-28 | 2026-09-25 | 10,900 | [received_time:int64, symbol:string, sum_open_interest:string, sum_open_interest_value:string, timestamp:int64] |
| f1c96e8c0c70 | parquet-rs version 55.2.0 | {} | 2026-09-25 | 2026-09-28 | 78 | [received_time:int64, symbol:string, sum_open_interest:string, sum_open_interest_value:string, timestamp:int64] |
| f1c96e8c0c70 | parquet-rs version 60.0.0 | {} | 2026-09-28 | 2026-10-31 | 799 | [received_time:int64, symbol:string, sum_open_interest:string, sum_open_interest_value:string, timestamp:int64] |

Row count read vs parquet footer mismatches: 0.

### mark_price

| schema_fp | created_by | kv_metadata | first_day | last_day | files | columns |
|---|---|---|---|---|---|---|
| 73d7b63f18d7 | parquet-rs version 55.1.0 | {} | 2025-06-28 | 2026-09-25 | 10,877 | [received_time:int64, event_time:int64, symbol:string, mark_price:string, index_price:string, estimated_settle_price:string, funding_rate:string, next_funding_time:int64] |
| 73d7b63f18d7 | parquet-rs version 55.2.0 | {} | 2026-09-25 | 2026-09-28 | 79 | [received_time:int64, event_time:int64, symbol:string, mark_price:string, index_price:string, estimated_settle_price:string, funding_rate:string, next_funding_time:int64] |
| 73d7b63f18d7 | parquet-rs version 60.0.0 | {} | 2026-09-28 | 2026-10-31 | 799 | [received_time:int64, event_time:int64, symbol:string, mark_price:string, index_price:string, estimated_settle_price:string, funding_rate:string, next_funding_time:int64] |

Row count read vs parquet footer mismatches: 0.


## 3. Trades (EXHAUSTIVE: every row, identity, NA rows, ids, clocks)

| month | files | rows | economic | NA | NA_nonzero | nonNA_nonpos | dup_id_rows | id_conflicts | missing_ids_in_file | id_gaps_in_file | id_inversions | recv_regress | delay_p50_ms | delay_p99_ms | delay_max_ms | neg_delay | synthetic_recv | offtick | recv_gaps_gt5s | recv_outside_hour | price_inexact | qty_inexact |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2025-06 | 67 | 9,216,283 | 9,214,401 | 1,877 | 0 | 5 | 5,411,452 | 0 | 14,112 | 7 | 624,924 | 295,229 | 128.969 | 145.602 | 56,971.771 | 1,141 | 0 | 0 | 58 | 0 | 0 | 0 |
| 2025-07 | 744 | 77,265,240 | 77,245,879 | 12,785 | 0 | 6,576 | 0 | 0 | 136,961 | 73 | 0 | 0 | 128.568 | 136.524 | 62,396.578 | 190 | 0 | 0 | 175 | 0 | 0 | 0 |
| 2025-08 | 744 | 82,661,358 | 82,515,541 | 145,227 | 0 | 590 | 0 | 0 | 325,839 | 104 | 0 | 0 | 124.166 | 137.613 | 87,550.548 | 0 | 0 | 0 | 99 | 0 | 0 | 0 |
| 2025-09 | 720 | 68,590,349 | 68,313,614 | 276,619 | 0 | 116 | 0 | 0 | 220,724 | 133 | 0 | 0 | 128.900 | 220.542 | 90,590.334 | 0 | 0 | 0 | 382 | 0 | 0 | 0 |
| 2025-10 | 744 | 133,186,271 | 132,931,842 | 249,991 | 0 | 4,438 | 0 | 0 | 3,216,923 | 566 | 0 | 0 | 128.761 | 261.118 | 78,025.667 | 0 | 0 | 0 | 202 | 0 | 0 | 0 |
| 2025-11 | 720 | 147,189,020 | 146,936,206 | 247,948 | 0 | 4,866 | 0 | 0 | 1,449,058 | 173 | 0 | 0 | 128.513 | 257.233 | 81,120.403 | 51 | 0 | 0 | 282 | 0 | 0 | 0 |
| 2025-12 | 744 | 109,875,396 | 109,665,680 | 209,359 | 0 | 357 | 0 | 0 | 350,022 | 149 | 0 | 0 | 126.542 | 150.526 | 114,835.305 | 0 | 0 | 0 | 516 | 0 | 0 | 0 |
| 2026-01 | 744 | 100,957,382 | 100,734,686 | 220,324 | 0 | 2,372 | 0 | 0 | 672,871 | 171 | 0 | 0 | 128.603 | 174.943 | 89,216.939 | 27 | 0 | 0 | 760 | 0 | 0 | 0 |
| 2026-02 | 672 | 190,370,071 | 190,202,530 | 166,696 | 0 | 845 | 0 | 0 | 326,771 | 147 | 0 | 0 | 143.165 | 158.434 | 83,236.335 | 381 | 0 | 0 | 120 | 0 | 0 | 0 |
| 2026-03 | 744 | 154,588,802 | 154,409,061 | 179,122 | 0 | 619 | 0 | 0 | 1,901,954 | 343 | 0 | 0 | 128.928 | 181.351 | 99,628.866 | 7 | 0 | 0 | 563 | 0 | 0 | 0 |
| 2026-04 | 720 | 103,624,049 | 103,439,910 | 184,046 | 0 | 93 | 0 | 0 | 757,696 | 211 | 0 | 0 | 129.268 | 234.110 | 126,254.749 | 10 | 0 | 0 | 464 | 0 | 0 | 0 |
| 2026-05 | 744 | 86,500,743 | 86,298,863 | 201,848 | 0 | 32 | 0 | 0 | 1,773,415 | 357 | 1 | 1 | 124.503 | 441.777 | 42,805.082 | 51 | 0 | 0 | 482 | 0 | 0 | 0 |
| 2026-06 | 720 | 146,405,975 | 146,216,068 | 189,241 | 0 | 666 | 0 | 0 | 514,309 | 110 | 0 | 0 | 127.271 | 155.676 | 66,990.647 | 0 | 0 | 0 | 110 | 0 | 0 | 0 |
| 2026-07 | 744 | 91,650,714 | 91,468,770 | 181,146 | 0 | 798 | 0 | 0 | 419,705 | 101 | 0 | 0 | 129.590 | 164.476 | 244,551.858 | 2,853 | 0 | 0 | 362 | 0 | 0 | 0 |
| 2026-08 | 744 | 92,869,470 | 92,559,994 | 302,091 | 0 | 7,385 | 0 | 0 | 229,744 | 80 | 0 | 0 | 126.804 | 215.500 | 35,791.805 | 426 | 0 | 0 | 885 | 1 | 0 | 0 |
| 2026-09 | 720 | 96,764,884 | 96,374,103 | 390,329 | 0 | 452 | 0 | 0 | 82,131 | 55 | 0 | 0 | 126.725 | 222.504 | 23,813.193 | 806 | 0 | 0 | 750 | 0 | 0 | 0 |
| 2026-10 | 744 | 88,484,962 | 88,160,818 | 323,988 | 0 | 156 | 0 | 0 | 342,469 | 57 | 0 | 0 | 126.723 | 153.688 | 151,473.374 | 0 | 0 | 0 | 644 | 0 | 0 | 0 |


Null counts over all files: `{}`. Order types: `{'MARKET': 1776687966, 'NA': 3482637, 'INSURANCE_FUND': 30366}`.


Canonical stream build (global dedup across files, availability order, continuity):

```
{
 "rows": 1780200969,
 "copies_dropped": 5411452,
 "cross_file_dups": 0,
 "na": 3481479,
 "nonpos": 30364,
 "conflicts": 0,
 "economic": 1771277674,
 "synthetic_recv": 0,
 "id_gaps": 2846,
 "missing_ids": 12799476,
 "missing_files": 5,
 "inversions": 2,
 "segments": 2847
}
```


Trade-id continuity gaps (missing ids after counting NA ids) by month — each breaks the bar segment and invalidates windows that intersect it:

| month | id_gaps | missing_ids | gap_span_s | max_span_s |
|---|---|---|---|---|
| 2025-06 | 7 | 14,112 | 364.350 | 72.889 |
| 2025-07 | 73 | 136,961 | 595.062 | 72.823 |
| 2025-08 | 104 | 325,839 | 371.264 | 68.368 |
| 2025-09 | 133 | 220,724 | 789.650 | 146.005 |
| 2025-10 | 568 | 3,241,661 | 5,212.829 | 259.409 |
| 2025-11 | 174 | 1,454,561 | 4,482.584 | 167.643 |
| 2025-12 | 149 | 350,022 | 940.149 | 114.051 |
| 2026-01 | 171 | 672,871 | 640.790 | 73.409 |
| 2026-02 | 147 | 326,771 | 504.341 | 69.868 |
| 2026-03 | 345 | 1,907,024 | 10,891.214 | 752.935 |
| 2026-04 | 212 | 767,219 | 1,712.896 | 134.751 |
| 2026-05 | 358 | 1,775,803 | 1,886.377 | 104.044 |
| 2026-06 | 110 | 514,309 | 538.294 | 98.267 |
| 2026-07 | 102 | 434,750 | 1,505.069 | 980.198 |
| 2026-08 | 81 | 232,249 | 563.718 | 177.466 |
| 2026-09 | 55 | 82,131 | 1,291.834 | 201.345 |
| 2026-10 | 57 | 342,469 | 825.720 | 224.143 |


Largest trade gaps:

| from | to | missing | recv_span_s |
|---|---|---|---|
| 2026-03-26 20:11:50 | 2026-03-26 20:15:51 | 126,378 | 241.124 |
| 2025-10-29 18:39:17 | 2025-10-29 18:40:38 | 104,263 | 80.812 |
| 2025-10-29 18:35:39 | 2025-10-29 18:37:35 | 103,225 | 116.416 |
| 2026-03-01 01:09:36 | 2026-03-01 01:21:43 | 98,979 | 727.691 |
| 2025-10-24 12:30:12 | 2025-10-24 12:32:10 | 97,726 | 117.910 |
| 2025-11-03 15:28:03 | 2025-11-03 15:29:51 | 75,737 | 108.074 |
| 2025-10-29 18:40:40 | 2025-10-29 18:42:42 | 72,969 | 122.288 |
| 2025-10-29 18:42:43 | 2025-10-29 18:44:45 | 68,315 | 121.621 |
| 2026-03-24 17:05:02 | 2026-03-24 17:11:11 | 63,875 | 369.159 |
| 2026-05-17 23:41:01 | 2026-05-17 23:42:12 | 62,252 | 71.002 |
| 2025-10-21 14:57:08 | 2025-10-21 14:59:04 | 61,396 | 116.201 |
| 2025-11-03 15:30:50 | 2025-11-03 15:32:52 | 60,199 | 122.017 |
| 2025-11-04 20:17:06 | 2025-11-04 20:18:46 | 58,832 | 100.297 |
| 2025-11-04 17:17:02 | 2025-11-04 17:18:40 | 56,422 | 98.331 |
| 2026-03-06 17:36:27 | 2026-03-06 17:39:44 | 55,774 | 197.627 |


## 4. Order book (EXHAUSTIVE per-file pass: message reassembly, copies, sequence, snapshots, clocks)

Per-file chain statistics below follow reception order inside each file only (no buffering); the causal resolver used for reconstruction (section 9) re-orders by sequence ids with buffering and is the reference for continuity.

| month | files | rows | messages | copies_per_msg | copy_conflicts | snapshots | snap_synthetic_recv | hour_checkpoints | msg_interval_p50_ms | levels_p50 | delay_p50_ms | delay_p999_ms | delay_max_ms | neg_delay_msgs | breaks_recvorder | stale_recvorder | recv_gaps_gt5s | recv_gaps_gt60s | gap_time_s | inconsistent_msgs | dup_level_msgs | offtick_rows | zero_qty_rows | noop_msgs | txn_null_updates |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2025-06 | 67 | 423,947,018 | 2,356,950 | 2.623 | 0 | 665 | 665 | 0 | 102.000 | 51.000 | 125.748 | 210.140 | 56,950.956 | 371 | 4 | 4 | 8 | 5 | 372.678 | 0 | 0 | 0 | 72,756,587 | 0 | 0 |
| 2025-07 | 744 | 2,773,223,716 | 33,948,579 | 1.000 | 0 | 0 | 0 | 0 | 102.000 | 52.000 | 127.863 | 157.594 | 62,416.353 | 364 | 73 | 0 | 48 | 5 | 834.029 | 0 | 0 | 133 | 554,889,509 | 0 | 0 |
| 2025-08 | 744 | 3,204,052,770 | 51,621,801 | 1.000 | 0 | 0 | 0 | 0 | 52.000 | 37.000 | 123.379 | 192.263 | 91,711.455 | 0 | 105 | 0 | 28 | 1 | 1,470.837 | 0 | 0 | 0 | 579,557,190 | 0 | 0 |
| 2025-09 | 720 | 2,714,742,577 | 49,927,022 | 1.000 | 0 | 0 | 0 | 0 | 52.000 | 35.000 | 128.033 | 346.834 | 89,831.563 | 0 | 134 | 0 | 71 | 2 | 1,132.960 | 0 | 0 | 0 | 422,812,367 | 0 | 0 |
| 2025-10 | 744 | 3,785,177,201 | 51,598,841 | 1.000 | 0 | 0 | 0 | 0 | 52.000 | 41.000 | 127.781 | 375.995 | 85,910.920 | 0 | 566 | 0 | 126 | 34 | 4,888.442 | 0 | 0 | 0 | 702,377,853 | 0 | 0 |
| 2025-11 | 720 | 4,258,410,142 | 49,620,540 | 1.000 | 0 | 0 | 0 | 0 | 52.000 | 46.000 | 127.579 | 378.174 | 81,161.371 | 33 | 175 | 0 | 139 | 40 | 5,288.164 | 0 | 0 | 0 | 770,095,540 | 0 | 0 |
| 2025-12 | 744 | 3,496,660,038 | 51,059,964 | 1.000 | 0 | 0 | 0 | 0 | 52.000 | 34.000 | 125.169 | 169.542 | 114,891.414 | 0 | 149 | 0 | 114 | 5 | 1,471.836 | 0 | 0 | 0 | 547,272,739 | 0 | 0 |
| 2026-01 | 744 | 3,288,922,356 | 51,060,436 | 1.000 | 0 | 0 | 0 | 0 | 52.000 | 33.000 | 127.476 | 215.362 | 89,235.890 | 51 | 172 | 0 | 53 | 4 | 792.390 | 0 | 0 | 0 | 484,791,575 | 0 | 0 |
| 2026-02 | 672 | 4,500,255,531 | 46,406,392 | 1.000 | 0 | 0 | 0 | 0 | 52.000 | 44.000 | 142.353 | 151.319 | 83,254.321 | 122 | 148 | 0 | 46 | 3 | 618.147 | 0 | 0 | 0 | 725,122,380 | 0 | 0 |
| 2026-03 | 711 | 4,169,490,624 | 48,865,093 | 1.000 | 0 | 0 | 0 | 0 | 52.000 | 44.000 | 127.013 | 270.416 | 100,447.144 | 26 | 332 | 0 | 366 | 40 | 12,057.049 | 0 | 0 | 0 | 605,365,119 | 0 | 0 |
| 2026-04 | 552 | 2,574,413,444 | 37,923,920 | 1.000 | 0 | 28 | 0 | 0 | 52.000 | 36.000 | 127.505 | 275.743 | 54,332.802 | 74 | 10 | 0 | 24 | 12 | 2,252.124 | 0 | 0 | 0 | 328,389,074 | 0 | 0 |
| 2026-05 | 744 | 3,639,002,391 | 85,140,852 | 1.000 | 0 | 12,992 | 16 | 0 | 28.000 | 25.000 | 122.932 | 361.419 | 45,683.218 | 0 | 68 | 16 | 2,117 | 339 | 138,326.029 | 0 | 0 | 0 | 380,389,375 | 0 | 0 |
| 2026-06 | 720 | 4,999,099,983 | 96,432,555 | 1.000 | 0 | 139 | 7 | 0 | 26.000 | 26.000 | 125.560 | 142.932 | 72,254.355 | 0 | 0 | 0 | 40 | 16 | 6,995.842 | 0 | 0 | 0 | 607,962,851 | 0 | 0 |
| 2026-07 | 744 | 3,695,217,964 | 98,964,345 | 1.000 | 0 | 85 | 9 | 0 | 28.000 | 21.000 | 128.461 | 186.977 | 62,202.615 | 646 | 0 | 0 | 39 | 9 | 4,881.505 | 0 | 0 | 0 | 347,248,607 | 0 | 0 |
| 2026-08 | 744 | 3,563,400,140 | 100,001,639 | 1.000 | 0 | 100 | 30 | 21 | 26.000 | 19.000 | 124.835 | 280.083 | 65,452.894 | 0 | 0 | 0 | 29 | 5 | 1,706.831 | 0 | 0 | 0 | 378,221,873 | 0 | 0 |
| 2026-09 | 720 | 3,574,522,716 | 96,907,259 | 1.000 | 0 | 454 | 361 | 392 | 26.000 | 20.000 | 124.924 | 318.788 | 62,635.507 | 0 | 1 | 0 | 34 | 17 | 3,090.472 | 0 | 0 | 0 | 422,950,884 | 0 | 0 |
| 2026-10 | 744 | 3,987,026,911 | 100,744,916 | 1.000 | 0 | 880 | 0 | 744 | 26.000 | 22.000 | 124.886 | 149.878 | 54,581.221 | 0 | 23 | 0 | 64 | 22 | 5,227.388 | 0 | 0 | 0 | 448,721,378 | 0 | 0 |


Null counts: `{'transaction_time': 49287120, 'first_update_id': 76105120, 'prev_final_update_id': 76105120, 'last_update_id': 58571460402, 'order_count': 58269454399}`; event types `{'snapshot': 76105120, 'update': 58571460402}`; sides `{'ask': 26202703246, 'bid': 32444862276}`.


Snapshots (anchors) by month:

| month | snapshots | bid_levels_p50 | bid_levels_min | synthetic_recv | hour_checkpoint | connects_next_in_file | crossed | days_with_snapshot |
|---|---|---|---|---|---|---|---|---|
| 2025-06 | 665 | 500.000 | 500 | 665 | 0 | 0 | 0 | 3 |
| 2026-04 | 28 | 500.000 | 500 | 0 | 0 | 28 | 0 | 17 |
| 2026-05 | 12,992 | 500.000 | 500 | 16 | 0 | 8,136 | 0 | 31 |
| 2026-06 | 139 | 500.000 | 500 | 7 | 0 | 124 | 0 | 30 |
| 2026-07 | 85 | 500.000 | 500 | 9 | 0 | 70 | 0 | 31 |
| 2026-08 | 100 | 500.000 | 500 | 30 | 21 | 89 | 0 | 31 |
| 2026-09 | 454 | 20,799.500 | 500 | 361 | 392 | 443 | 0 | 30 |
| 2026-10 | 880 | 24,205.500 | 100 | 0 | 744 | 849 | 0 | 31 |


Longest intra-file reception gaps of book messages:

| from | dur_s |
|---|---|
| 2026-07-13 06:36:53 | 1,267.613 |
| 2025-08-29 06:18:03 | 1,141.646 |
| 2026-03-26 05:06:41 | 752.748 |
| 2026-07-24 05:26:55 | 747.437 |
| 2026-03-01 01:09:36 | 727.511 |
| 2026-03-19 00:35:05 | 689.384 |
| 2026-07-24 06:41:46 | 675.740 |
| 2026-06-02 14:24:02 | 657.759 |
| 2026-05-24 20:09:09 | 651.238 |
| 2026-05-25 22:08:15 | 645.429 |
| 2026-05-25 09:25:15 | 644.278 |
| 2026-05-25 20:17:16 | 643.555 |
| 2026-05-27 15:03:17 | 643.063 |
| 2026-05-26 13:03:17 | 642.629 |
| 2026-05-28 04:03:05 | 641.126 |
| 2026-05-25 02:33:19 | 640.683 |
| 2026-05-25 00:06:20 | 640.248 |
| 2026-07-15 00:10:21 | 639.040 |
| 2026-05-07 08:05:32 | 616.021 |
| 2026-10-22 17:32:31 | 615.848 |


## 5. Mark price / index / funding (EXHAUSTIVE)

| month | files | rows | interval_p50_ms | dup_event_rows | evt_conflicts | delay_p50_ms | delay_max_ms | delay_gt3s | neg_delay | evt_gaps_gt3s | evt_gaps_gt60s | evt_gap_time_s | basis_min_bps | basis_max_bps | funding_min | funding_max |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2025-06 | 67 | 632,454 | 0.000 | 391,660 | 0 | 184.875 | 55,898.560 | 65 | 34 | 6 | 5 | 411.996 | -7.078 | 0.578 | -0.000 | 0.000 |
| 2025-07 | 744 | 2,676,925 | 1,000.000 | 0 | 0 | 189.242 | 568,335.275 | 1,868 | 30 | 38 | 7 | 1,428.996 | -11.157 | 19.942 | -0.000 | 0.000 |
| 2025-08 | 744 | 2,676,565 | 1,000.000 | 0 | 0 | 186.622 | 241,460.645 | 2,794 | 0 | 35 | 11 | 1,825.996 | -22.664 | 7.817 | -0.000 | 0.000 |
| 2025-09 | 720 | 2,588,822 | 1,000.000 | 0 | 0 | 185.489 | 352,416.076 | 2,875 | 0 | 63 | 16 | 3,166.003 | -18.383 | 7.252 | -0.000 | 0.000 |
| 2025-10 | 744 | 2,669,340 | 1,000.000 | 0 | 0 | 190.362 | 107,754.432 | 13,846 | 0 | 121 | 63 | 8,953.034 | -69.163 | 55.474 | -0.000 | 0.000 |
| 2025-11 | 720 | 2,582,725 | 1,000.000 | 0 | 0 | 196.874 | 119,979.746 | 16,124 | 0 | 103 | 62 | 9,303.005 | -50.554 | 5.508 | -0.000 | 0.000 |
| 2025-12 | 744 | 2,674,601 | 1,000.000 | 0 | 0 | 194.465 | 97,391.596 | 5,076 | 0 | 78 | 27 | 3,764.037 | -16.867 | 6.893 | -0.000 | 0.000 |
| 2026-01 | 744 | 2,675,769 | 1,000.000 | 0 | 0 | 196.986 | 78,585.032 | 5,023 | 0 | 55 | 10 | 2,632.009 | -40.416 | 21.715 | -0.000 | 0.000 |
| 2026-02 | 672 | 2,417,808 | 1,000.000 | 0 | 0 | 217.179 | 69,641.254 | 841 | 2 | 36 | 8 | 1,383.005 | -17.878 | 7.774 | -0.000 | 0.000 |
| 2026-03 | 744 | 2,667,985 | 1,000.000 | 0 | 0 | 209.894 | 342,993.273 | 4,827 | 0 | 239 | 43 | 9,505.055 | -15.617 | 14.929 | -0.000 | 0.000 |
| 2026-04 | 696 | 2,500,648 | 1,000.000 | 0 | 0 | 209.546 | 80,383.525 | 13,127 | 0 | 52 | 15 | 2,071.991 | -15.404 | 6.658 | -0.000 | 0.000 |
| 2026-05 | 744 | 2,676,915 | 1,000.000 | 0 | 0 | 218.462 | 83,976.661 | 2,304 | 2 | 72 | 9 | 1,366.981 | -20.114 | 5.335 | -0.000 | 0.000 |
| 2026-06 | 720 | 2,590,271 | 1,000.000 | 0 | 0 | 218.255 | 104,157.602 | 31,270 | 0 | 55 | 12 | 1,663.004 | -21.428 | 12.763 | -0.000 | 0.000 |
| 2026-07 | 744 | 2,675,744 | 1,000.000 | 0 | 0 | 225.584 | 94,579.612 | 39,091 | 27 | 76 | 13 | 2,657.994 | -10.888 | 7.018 | -0.000 | 0.000 |
| 2026-08 | 744 | 2,630,859 | 1,000.000 | 0 | 0 | 224.513 | 235,374.438 | 109,657 | 26 | 854 | 340 | 48,238.080 | -16.034 | 38.100 | -0.000 | 0.000 |
| 2026-09 | 720 | 2,532,652 | 1,000.000 | 0 | 0 | 206.646 | 178,312.928 | 81,650 | 30 | 911 | 506 | 60,193.861 | -26.813 | 11.168 | -0.000 | 0.000 |
| 2026-10 | 744 | 2,676,173 | 1,000.000 | 0 | 0 | 202.128 | 57,600.628 | 1,472 | 0 | 82 | 6 | 2,228.006 | -15.731 | 4.845 | -0.000 | 0.000 |


Null counts: `{}`.


Mark-price market-clock gaps ≥30 s by month and how many start at minute 01–05 of the hour (vendor-declared hh:02–05 outages since mid-Aug 2026):

| month | gaps_ge30s | starting_hh01_05 | dur_p50_s |
|---|---|---|---|
| 2025-06 | 6 | 0 | 70.498 |
| 2025-07 | 21 | 3 | 53.001 |
| 2025-08 | 24 | 5 | 56.502 |
| 2025-09 | 46 | 15 | 55.500 |
| 2025-10 | 103 | 66 | 68.000 |
| 2025-11 | 89 | 50 | 94.002 |
| 2025-12 | 60 | 21 | 56.500 |
| 2026-01 | 41 | 13 | 53.996 |
| 2026-02 | 21 | 0 | 55.008 |
| 2026-03 | 66 | 8 | 64.001 |
| 2026-04 | 26 | 7 | 66.499 |
| 2026-05 | 14 | 4 | 62.500 |
| 2026-06 | 24 | 9 | 59.498 |
| 2026-07 | 36 | 14 | 49.001 |
| 2026-08 | 775 | 439 | 56.002 |
| 2026-09 | 856 | 638 | 66.000 |
| 2026-10 | 27 | 9 | 37.000 |

## 6. Ticker (EXHAUSTIVE)

| month | files | rows | interval_p50_ms | window_min_s | window_max_s | price_change_mismatch | pct_mismatch | vwap_outside_hl | last_outside_hl | dup_event_rows | evt_conflicts | delay_p50_ms | recv_gaps_gt10s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2025-06 | 67 | 589,095 | 0.000 | 86,400.000 | 86,459.999 | 0 | 0 | 0 | 0 | 364,203 | 21 | 306.971 | 8 |
| 2025-07 | 744 | 2,591,634 | 1,007.000 | 86,400.000 | 86,459.999 | 0 | 0 | 0 | 0 | 77 | 77 | 265.057 | 31 |
| 2025-08 | 744 | 2,596,276 | 1,007.000 | 85,261.024 | 86,459.999 | 0 | 0 | 0 | 0 | 83 | 83 | 269.049 | 62 |
| 2025-09 | 720 | 2,473,239 | 1,008.000 | 86,400.000 | 86,459.999 | 0 | 0 | 0 | 0 | 89 | 89 | 287.995 | 88 |
| 2025-10 | 744 | 2,620,767 | 1,004.000 | 86,400.000 | 86,459.999 | 0 | 0 | 0 | 0 | 49 | 49 | 257.678 | 91 |
| 2025-11 | 720 | 2,513,716 | 1,003.000 | 86,400.000 | 86,459.999 | 0 | 0 | 0 | 0 | 71 | 71 | 251.425 | 124 |
| 2025-12 | 744 | 2,539,359 | 1,008.500 | 86,400.000 | 86,459.999 | 0 | 0 | 0 | 0 | 96 | 96 | 276.686 | 91 |
| 2026-01 | 744 | 2,496,974 | 1,011.250 | 86,400.000 | 86,459.999 | 0 | 0 | 0 | 0 | 138 | 138 | 303.968 | 87 |
| 2026-02 | 672 | 2,338,612 | 1,005.000 | 86,400.000 | 86,459.999 | 0 | 0 | 0 | 0 | 113 | 113 | 279.092 | 41 |
| 2026-03 | 744 | 2,579,689 | 1,005.250 | 86,400.000 | 86,459.999 | 0 | 0 | 0 | 0 | 131 | 131 | 271.248 | 276 |
| 2026-04 | 696 | 1,808,447 | 1,123.000 | 86,400.000 | 86,459.999 | 0 | 0 | 0 | 0 | 91 | 91 | 290.100 | 24 |
| 2026-05 | 744 | 2,511,500 | 1,012.000 | 86,400.000 | 86,459.999 | 0 | 0 | 0 | 0 | 167 | 167 | 387.186 | 26 |
| 2026-06 | 720 | 2,500,718 | 1,005.250 | 86,400.000 | 86,459.999 | 0 | 0 | 0 | 0 | 113 | 113 | 270.710 | 10 |
| 2026-07 | 744 | 2,542,458 | 1,007.000 | 86,400.000 | 86,459.999 | 0 | 0 | 0 | 0 | 185 | 185 | 300.952 | 34 |
| 2026-08 | 744 | 2,488,098 | 1,010.250 | 86,400.000 | 86,459.999 | 0 | 0 | 0 | 0 | 188 | 188 | 312.763 | 45 |
| 2026-09 | 720 | 2,402,281 | 1,011.250 | 86,400.000 | 86,459.999 | 0 | 0 | 0 | 0 | 132 | 132 | 311.984 | 44 |
| 2026-10 | 744 | 2,503,784 | 1,009.000 | 84,780.184 | 86,459.999 | 0 | 0 | 0 | 0 | 1 | 1 | 306.858 | 26 |


Null counts: `{}`.

## 7. Open interest (EXHAUSTIVE)

| month | files | rows | unique_ts_in_files | ts_not_5min | poll_batches | batch_interval_p50_s | delay_min_s | delay_p50_s | delay_max_s | points_before_file_hour |
|---|---|---|---|---|---|---|---|---|---|---|
| 2025-06 | 66 | 1,290 | 1,290 | 0 | 484 | 490.346 | 16.537 | 336.316 | 149,810.965 | 538 |
| 2025-07 | 744 | 8,918 | 8,918 | 0 | 5,416 | 494.342 | 7.567 | 336.912 | 1,027.189 | 451 |
| 2025-08 | 744 | 8,926 | 8,926 | 0 | 5,393 | 496.349 | 4.373 | 339.320 | 1,074.663 | 449 |
| 2025-09 | 720 | 8,640 | 8,640 | 0 | 5,174 | 496.350 | 2.801 | 341.123 | 955.173 | 381 |
| 2025-10 | 744 | 8,927 | 8,927 | 0 | 5,206 | 514.387 | 1.486 | 346.345 | 693.155 | 469 |
| 2025-11 | 720 | 8,640 | 8,640 | 0 | 4,976 | 514.413 | 6.275 | 348.757 | 709.278 | 483 |
| 2025-12 | 744 | 8,929 | 8,929 | 0 | 5,050 | 530.384 | 4.149 | 357.073 | 708.669 | 498 |
| 2026-01 | 744 | 8,927 | 8,927 | 0 | 5,044 | 530.383 | 4.242 | 356.528 | 709.036 | 504 |
| 2026-02 | 672 | 8,065 | 8,065 | 0 | 4,527 | 534.374 | 4.550 | 357.835 | 1,665.990 | 462 |
| 2026-03 | 744 | 8,927 | 8,927 | 0 | 5,031 | 532.400 | 1.432 | 356.889 | 710.489 | 494 |
| 2026-04 | 720 | 8,641 | 8,641 | 0 | 4,876 | 532.375 | 3.111 | 355.496 | 708.551 | 478 |
| 2026-05 | 744 | 8,928 | 8,928 | 0 | 5,069 | 528.371 | 2.730 | 353.640 | 1,088.470 | 495 |
| 2026-06 | 720 | 8,639 | 8,639 | 0 | 4,904 | 528.372 | 3.575 | 355.264 | 700.967 | 485 |
| 2026-07 | 744 | 8,929 | 8,929 | 0 | 5,050 | 530.373 | 2.408 | 355.954 | 835.463 | 495 |
| 2026-08 | 744 | 8,928 | 8,928 | 0 | 4,664 | 594.430 | 7.830 | 376.961 | 765.982 | 553 |
| 2026-09 | 720 | 8,640 | 8,640 | 0 | 4,267 | 606.591 | 1.738 | 394.377 | 3,089.663 | 606 |
| 2026-10 | 744 | 21,034 | 21,034 | 2,768 | 4,337 | 616.480 | 1.442 | 400.877 | 152,809.451 | 12,386 |


Null counts: `{}`.


Repair markers in OI parquet metadata:

| s_repairs | files | first | last |
|---|---|---|---|
| ["binance_futures_oi_cross_hour_dedup/v1"] | 10,900 | 2025-06-28 | 2026-09-25 |
|  | 878 | 2025-06-28 | 2026-10-31 |


Canonical OI (versions by first receipt, cross-file repeats removed):

```
{"rows_on_grid": 151160, "quarantined_offgrid": 2768, "versions": 141927, "unique_ts": 141821, "ts_with_conflicting_versions": 47, "revised_ts": 106, "copies_dropped": 9233, "ts_not_5m": 0}
```

## 8. Liquidations (EXHAUSTIVE)

| month | files | rows | filled_eq_qty | filled_lt_qty | lastfilled_lt_filled | dup_exact | dup_except_recv | same_order_key | notional_usdt | delay_p50_ms | delay_max_ms |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2025-06 | 66 | 1,767 | 1,767 | 0 | 268 | 0 | 0 | 0 | 30,995,583 | 124.120 | 5,017.833 |
| 2025-07 | 742 | 33,714 | 33,714 | 0 | 5,192 | 0 | 0 | 0 | 525,438,186 | 128.711 | 58,358.084 |
| 2025-08 | 743 | 32,684 | 32,684 | 0 | 5,097 | 0 | 0 | 0 | 348,731,983 | 128.317 | 64,712.709 |
| 2025-09 | 720 | 26,102 | 26,102 | 0 | 4,060 | 0 | 0 | 0 | 275,577,130 | 128.026 | 67,326.318 |
| 2025-10 | 743 | 44,925 | 44,925 | 0 | 8,854 | 0 | 0 | 0 | 728,165,124 | 128.999 | 106,274.435 |
| 2025-11 | 720 | 52,640 | 52,640 | 0 | 11,182 | 0 | 0 | 0 | 561,886,767 | 129.120 | 112,553.319 |
| 2025-12 | 742 | 33,463 | 33,463 | 0 | 6,483 | 0 | 0 | 0 | 358,368,355 | 127.864 | 97,935.355 |
| 2026-01 | 738 | 29,870 | 29,870 | 0 | 5,472 | 0 | 0 | 0 | 373,518,693 | 129.156 | 77,558.225 |
| 2026-02 | 672 | 50,567 | 50,567 | 0 | 13,305 | 0 | 0 | 0 | 554,497,888 | 142.627 | 61,638.612 |
| 2026-03 | 744 | 40,839 | 40,839 | 0 | 8,487 | 0 | 0 | 0 | 386,500,665 | 129.484 | 246,162.597 |
| 2026-04 | 694 | 28,467 | 28,467 | 0 | 7,038 | 0 | 0 | 0 | 763,940,406 | 128.511 | 29,326.074 |
| 2026-05 | 732 | 25,304 | 25,304 | 0 | 6,881 | 0 | 0 | 0 | 1,025,885,760 | 122.963 | 1,228.911 |
| 2026-06 | 720 | 44,150 | 44,150 | 0 | 14,927 | 0 | 0 | 0 | 2,000,289,217 | 125.994 | 1,568.086 |
| 2026-07 | 743 | 25,559 | 25,559 | 0 | 7,107 | 0 | 0 | 0 | 761,668,005 | 129.516 | 21,028.803 |
| 2026-08 | 737 | 28,466 | 28,466 | 0 | 8,637 | 0 | 0 | 0 | 1,379,822,595 | 125.613 | 11,876.529 |
| 2026-09 | 717 | 29,699 | 29,699 | 0 | 8,796 | 0 | 0 | 0 | 1,131,409,244 | 125.597 | 20,904.134 |
| 2026-10 | 736 | 27,085 | 27,085 | 0 | 7,694 | 0 | 0 | 0 | 1,274,274,422 | 124.983 | 169,426.028 |


Null counts: `{}`.


Values: status `{'FILLED': 555301}`, type `{'LIMIT': 555301}`, tif `{'IOC': 555301}`, side `{'BUY': 254622, 'SELL': 300679}`.


Canonical liquidations:

```
{"rows": 555301, "kept": 555301, "exact_copies_dropped": 0, "maybe_same_order_rows": 0, "hours_with_file": 11709, "non_filled": 0}
```
