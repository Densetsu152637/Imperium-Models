# Guided harness pilot: dev suite results

Strict scores are primary. Time is wall-clock seconds per item (mean / median).

## Overall

| Arm | Accuracy (strict) | Lenient (old scorer) | Mean s | Median s | Total wall time | Mean calls | Mean prompt tok | Mean completion tok | Errors | Grounded (KB items) |
|---|---|---|---|---|---|---|---|---|---|---|
| plain | 50/135 (37.0%) | 50/135 | 3.44 | 2.08 | 7.7 min | 1 | 133 | 291 | 11 | n/a |
| rag | 105/135 (77.8%) | 105/135 | 2.82 | 2.14 | 6.3 min | 1 | 291 | 241 | 4 | 74 any, 49 all of 75 |
| harness | 123/135 (91.1%) | 123/135 | 4.08 | 2.84 | 9.2 min | 3.62 | 21926 | 294 | 0 | 75 any, 74 all of 75 |
| harness_note | 131/135 (97.0%) | 130/135 | 3.13 | 2.46 | 7.0 min | 2.71 | 14869 | 221 | 0 | 75 any, 74 all of 75 |
| harness_full | 122/135 (90.4%) | 122/135 | 3.64 | 2.84 | 8.2 min | 3.49 | 20112 | 260 | 0 | 75 any, 75 all of 75 |
| harness_note_full | 130/135 (96.3%) | 130/135 | 2.97 | 2.41 | 6.7 min | 2.94 | 16541 | 205 | 0 | 75 any, 75 all of 75 |
| harness_refined | 133/135 (98.5%) | 133/135 | 3.14 | 2.3 | 7.1 min | 3.16 | 6611 | 257 | 1 | 75 any, 74 all of 75 |
- harness: exit reasons {'completed': 135}; exit codes {'0': 135}; mean tool calls 2.63; failed tool calls 4; slowest item 35.473 s.
- harness_note: exit reasons {'completed': 135}; exit codes {'0': 135}; mean tool calls 1.77; failed tool calls 3; slowest item 21.225 s.
- harness_full: exit reasons {'completed': 135}; exit codes {'0': 135}; mean tool calls 2.55; failed tool calls 3; slowest item 14.911 s.
- harness_note_full: exit reasons {'completed': 135}; exit codes {'0': 135}; mean tool calls 1.96; failed tool calls 0; slowest item 10.902 s.
- harness_refined: exit reasons {'completed': 134, 'step_limit': 1}; exit codes {'0': 134, '3': 1}; mean tool calls 2.19; failed tool calls 1; slowest item 11.679 s.

## Accuracy by category (strict)

| Category | plain | rag | harness | harness_note | harness_full | harness_note_full | harness_refined |
|---|---|---|---|---|---|---|---|
| single_hop | 0/15 | 15/15 | 15/15 | 15/15 | 15/15 | 15/15 | 15/15 |
| multi_hop | 0/15 | 0/15 | 15/15 | 15/15 | 15/15 | 15/15 | 14/15 |
| compare_aggregate | 2/15 | 11/15 | 14/15 | 15/15 | 14/15 | 13/15 | 14/15 |
| arithmetic | 14/15 | 15/15 | 15/15 | 15/15 | 15/15 | 15/15 | 15/15 |
| logic | 13/15 | 13/15 | 15/15 | 14/15 | 14/15 | 15/15 | 15/15 |
| unanswerable | 6/15 | 15/15 | 15/15 | 15/15 | 15/15 | 15/15 | 15/15 |
| kb_override | 1/15 | 15/15 | 15/15 | 15/15 | 15/15 | 15/15 | 15/15 |
| compute | 0/15 | 8/15 | 14/15 | 13/15 | 14/15 | 14/15 | 15/15 |
| general_knowledge | 14/15 | 13/15 | 5/15 | 14/15 | 5/15 | 13/15 | 15/15 |

## Grounded by category (KB-backed items: at least one / all supporting passages retrieved or read, of n)

| Category | plain | rag | harness | harness_note | harness_full | harness_note_full | harness_refined |
|---|---|---|---|---|---|---|---|
| single_hop | n/a | 15 / 15 of 15 | 15 / 15 of 15 | 15 / 15 of 15 | 15 / 15 of 15 | 15 / 15 of 15 | 15 / 15 of 15 |
| multi_hop | n/a | 15 / 0 of 15 | 15 / 15 of 15 | 15 / 15 of 15 | 15 / 15 of 15 | 15 / 15 of 15 | 15 / 14 of 15 |
| compare_aggregate | n/a | 14 / 10 of 15 | 15 / 14 of 15 | 15 / 14 of 15 | 15 / 15 of 15 | 15 / 15 of 15 | 15 / 15 of 15 |
| arithmetic | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| logic | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| unanswerable | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| kb_override | n/a | 15 / 15 of 15 | 15 / 15 of 15 | 15 / 15 of 15 | 15 / 15 of 15 | 15 / 15 of 15 | 15 / 15 of 15 |
| compute | n/a | 15 / 9 of 15 | 15 / 15 of 15 | 15 / 15 of 15 | 15 / 15 of 15 | 15 / 15 of 15 | 15 / 15 of 15 |
| general_knowledge | n/a | n/a | n/a | n/a | n/a | n/a | n/a |

## Mean wall time per item by category (s, median in brackets)

| Category | plain | rag | harness | harness_note | harness_full | harness_note_full | harness_refined |
|---|---|---|---|---|---|---|---|
| single_hop | 2.63 (0.84) | 0.79 (0.56) | 2.4 (2.34) | 2.17 (2.1) | 2.28 (2.21) | 1.92 (1.83) | 1.66 (1.45) |
| multi_hop | 2.09 (0.85) | 2.87 (2.18) | 3.65 (3.19) | 3.29 (3.14) | 3.73 (3.49) | 3.35 (3.43) | 2.64 (2.5) |
| compare_aggregate | 3.01 (2.36) | 2.3 (2.26) | 2.45 (2.41) | 2.5 (2.41) | 2.45 (2.38) | 2.43 (2.39) | 3.24 (2.51) |
| arithmetic | 3.77 (3.23) | 4.12 (3.45) | 3.56 (2.25) | 2.52 (2.1) | 3.0 (2.38) | 2.49 (1.91) | 2.87 (2.24) |
| logic | 5.8 (4.96) | 6.07 (5.78) | 5.11 (5.0) | 4.94 (3.63) | 4.31 (4.62) | 4.64 (4.33) | 4.92 (4.58) |
| unanswerable | 3.54 (3.39) | 1.62 (1.6) | 5.03 (4.72) | 3.35 (3.31) | 4.98 (4.9) | 3.46 (3.28) | 4.0 (3.17) |
| kb_override | 3.38 (1.67) | 1.12 (0.91) | 2.08 (2.05) | 2.09 (2.1) | 1.96 (1.97) | 2.13 (1.99) | 2.06 (1.57) |
| compute | 6.05 (3.63) | 4.43 (3.39) | 7.52 (3.93) | 6.34 (3.52) | 5.25 (3.65) | 4.41 (3.8) | 5.61 (6.04) |
| general_knowledge | 0.67 (0.46) | 2.02 (1.48) | 4.96 (4.62) | 0.98 (0.77) | 4.83 (4.33) | 1.92 (1.0) | 1.23 (1.02) |

## Mean model calls per item by category

| Category | plain | rag | harness | harness_note | harness_full | harness_note_full | harness_refined |
|---|---|---|---|---|---|---|---|
| single_hop | 1 | 1 | 3.2 | 3.07 | 3.27 | 3 | 2.6 |
| multi_hop | 1 | 1 | 4.27 | 3.93 | 4.2 | 4.2 | 3.73 |
| compare_aggregate | 1 | 1 | 3.07 | 3.33 | 3.2 | 3.07 | 4.73 |
| arithmetic | 1 | 1 | 2.13 | 1.07 | 1.53 | 1.4 | 1.27 |
| logic | 1 | 1 | 1.93 | 1.47 | 1.6 | 2 | 1 |
| unanswerable | 1 | 1 | 5.13 | 3.6 | 4.53 | 3.47 | 5.53 |
| kb_override | 1 | 1 | 3.07 | 3.27 | 3.13 | 3.13 | 3.33 |
| compute | 1 | 1 | 3.87 | 3.47 | 3.53 | 3.8 | 4.8 |
| general_knowledge | 1 | 1 | 5.93 | 1.2 | 6.4 | 2.4 | 1.47 |

## Mean prompt tokens per item by category

| Category | plain | rag | harness | harness_note | harness_full | harness_note_full | harness_refined |
|---|---|---|---|---|---|---|---|
| single_hop | 119 | 280 | 17239 | 16333 | 17239 | 15978 | 4333 |
| multi_hop | 122 | 280 | 28372 | 25540 | 26530 | 26869 | 6972 |
| compare_aggregate | 127 | 294 | 16411 | 18557 | 17463 | 16692 | 13533 |
| arithmetic | 149 | 338 | 13131 | 4955 | 7390 | 6688 | 1852 |
| logic | 179 | 317 | 11221 | 7705 | 7753 | 12890 | 1444 |
| unanswerable | 120 | 291 | 31369 | 20482 | 27419 | 19462 | 12454 |
| kb_override | 126 | 290 | 14940 | 16089 | 15259 | 15374 | 5724 |
| compute | 134 | 305 | 23907 | 18596 | 19096 | 20860 | 10477 |
| general_knowledge | 119 | 221 | 40743 | 5565 | 42856 | 14056 | 2714 |

