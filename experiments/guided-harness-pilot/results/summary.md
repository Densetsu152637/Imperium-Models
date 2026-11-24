# Guided harness pilot: results

Accuracy is n correct / n. Time is wall-clock seconds per item (mean / median).

## Overall

| Arm | Accuracy | Mean s | Median s | Total wall time | Mean calls | Mean prompt tok | Mean completion tok | Errors |
|---|---|---|---|---|---|---|---|---|
| plain | 50/135 (37.0%) | 3.44 | 2.08 | 7.7 min | 1 | 133 | 291 | 11 |
| rag | 105/135 (77.8%) | 2.82 | 2.14 | 6.3 min | 1 | 291 | 241 | 4 |

## Accuracy by category

| Category | plain | rag |
|---|---|---|
| single_hop | 0/15 | 15/15 |
| multi_hop | 0/15 | 0/15 |
| compare_aggregate | 2/15 | 11/15 |
| arithmetic | 14/15 | 15/15 |
| logic | 13/15 | 13/15 |
| unanswerable | 6/15 | 15/15 |
| kb_override | 1/15 | 15/15 |
| compute | 0/15 | 8/15 |
| general_knowledge | 14/15 | 13/15 |

## Mean wall time per item by category (s)

| Category | plain | rag |
|---|---|---|
| single_hop | 2.63 (median 0.84) | 0.79 (median 0.56) |
| multi_hop | 2.09 (median 0.85) | 2.87 (median 2.18) |
| compare_aggregate | 3.01 (median 2.36) | 2.3 (median 2.26) |
| arithmetic | 3.77 (median 3.23) | 4.12 (median 3.45) |
| logic | 5.8 (median 4.96) | 6.07 (median 5.78) |
| unanswerable | 3.54 (median 3.39) | 1.62 (median 1.6) |
| kb_override | 3.38 (median 1.67) | 1.12 (median 0.91) |
| compute | 6.05 (median 3.63) | 4.43 (median 3.39) |
| general_knowledge | 0.67 (median 0.46) | 2.02 (median 1.48) |

## Mean model calls per item by category

| Category | plain | rag |
|---|---|---|
| single_hop | 1 | 1 |
| multi_hop | 1 | 1 |
| compare_aggregate | 1 | 1 |
| arithmetic | 1 | 1 |
| logic | 1 | 1 |
| unanswerable | 1 | 1 |
| kb_override | 1 | 1 |
| compute | 1 | 1 |
| general_knowledge | 1 | 1 |
