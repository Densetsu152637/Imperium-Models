# Guided harness pilot: results

Accuracy is n correct / n. Time is wall-clock seconds per item (mean / median).

## Overall

| Arm | Accuracy | Mean s | Median s | Total wall time | Mean calls | Mean prompt tok | Mean completion tok | Errors |
|---|---|---|---|---|---|---|---|---|
| plain | 50/135 (37.0%) | 3.44 | 2.08 | 7.7 min | 1 | 133 | 291 | 11 |
| rag | 105/135 (77.8%) | 2.82 | 2.14 | 6.3 min | 1 | 291 | 241 | 4 |
| harness | 123/135 (91.1%) | 4.08 | 2.84 | 9.2 min | 3.62 | 21926 | 294 | 0 |
| harness_note | 130/135 (96.3%) | 3.13 | 2.46 | 7.0 min | 2.71 | 14869 | 221 | 0 |

Harness (Imperium agent loop): exit reasons {'completed': 135}; exit codes {'0': 135}; mean tool calls per item 2.63; failed tool calls 4; slowest item 35.473 s.

## Accuracy by category

| Category | plain | rag | harness | harness_note |
|---|---|---|---|---|
| single_hop | 0/15 | 15/15 | 15/15 | 15/15 |
| multi_hop | 0/15 | 0/15 | 15/15 | 15/15 |
| compare_aggregate | 2/15 | 11/15 | 14/15 | 15/15 |
| arithmetic | 14/15 | 15/15 | 15/15 | 15/15 |
| logic | 13/15 | 13/15 | 15/15 | 14/15 |
| unanswerable | 6/15 | 15/15 | 15/15 | 15/15 |
| kb_override | 1/15 | 15/15 | 15/15 | 15/15 |
| compute | 0/15 | 8/15 | 14/15 | 13/15 |
| general_knowledge | 14/15 | 13/15 | 5/15 | 13/15 |

## Mean wall time per item by category (s)

| Category | plain | rag | harness | harness_note |
|---|---|---|---|---|
| single_hop | 2.63 (median 0.84) | 0.79 (median 0.56) | 2.4 (median 2.34) | 2.17 (median 2.1) |
| multi_hop | 2.09 (median 0.85) | 2.87 (median 2.18) | 3.65 (median 3.19) | 3.29 (median 3.14) |
| compare_aggregate | 3.01 (median 2.36) | 2.3 (median 2.26) | 2.45 (median 2.41) | 2.5 (median 2.41) |
| arithmetic | 3.77 (median 3.23) | 4.12 (median 3.45) | 3.56 (median 2.25) | 2.52 (median 2.1) |
| logic | 5.8 (median 4.96) | 6.07 (median 5.78) | 5.11 (median 5.0) | 4.94 (median 3.63) |
| unanswerable | 3.54 (median 3.39) | 1.62 (median 1.6) | 5.03 (median 4.72) | 3.35 (median 3.31) |
| kb_override | 3.38 (median 1.67) | 1.12 (median 0.91) | 2.08 (median 2.05) | 2.09 (median 2.1) |
| compute | 6.05 (median 3.63) | 4.43 (median 3.39) | 7.52 (median 3.93) | 6.34 (median 3.52) |
| general_knowledge | 0.67 (median 0.46) | 2.02 (median 1.48) | 4.96 (median 4.62) | 0.98 (median 0.77) |

## Mean model calls per item by category

| Category | plain | rag | harness | harness_note |
|---|---|---|---|---|
| single_hop | 1 | 1 | 3.2 | 3.07 |
| multi_hop | 1 | 1 | 4.27 | 3.93 |
| compare_aggregate | 1 | 1 | 3.07 | 3.33 |
| arithmetic | 1 | 1 | 2.13 | 1.07 |
| logic | 1 | 1 | 1.93 | 1.47 |
| unanswerable | 1 | 1 | 5.13 | 3.6 |
| kb_override | 1 | 1 | 3.07 | 3.27 |
| compute | 1 | 1 | 3.87 | 3.47 |
| general_knowledge | 1 | 1 | 5.93 | 1.2 |

## Harness mean tool calls per item by category

| Category | Tool calls |
|---|---|
| single_hop | 2.2 |
| multi_hop | 3.33 |
| compare_aggregate | 2.07 |
| arithmetic | 1.13 |
| logic | 0.93 |
| unanswerable | 4.13 |
| kb_override | 2.07 |
| compute | 2.87 |
| general_knowledge | 4.93 |
