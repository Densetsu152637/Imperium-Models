# Guided harness pilot: hard suite results

Strict scores are primary. Time is wall-clock seconds per item (mean / median).

## Overall

| Arm | Accuracy (strict) | Lenient (old scorer) | Mean s | Median s | Total wall time | Mean calls | Mean prompt tok | Mean completion tok | Errors | Grounded (KB items) |
|---|---|---|---|---|---|---|---|---|---|---|
| plain | 4/72 (5.6%) | 4/72 | 4.48 | 2.49 | 5.4 min | 1 | 134 | 413 | 11 | n/a |
| rag | 42/72 (58.3%) | 42/72 | 3.67 | 3.3 | 4.4 min | 1 | 310 | 336 | 3 | 65 any, 38 all of 67 |
| harness | 65/72 (90.3%) | 65/72 | 6.27 | 3.95 | 7.5 min | 5.31 | 34663 | 474 | 3 | 66 any, 64 all of 67 |
| harness_note | 70/72 (97.2%) | 70/72 | 5.87 | 4.18 | 7.0 min | 5.04 | 33059 | 442 | 1 | 67 any, 66 all of 67 |
| harness_refined | 67/72 (93.1%) | 67/72 | 8.6 | 5.15 | 10.3 min | 6.33 | 20028 | 728 | 1 | 66 any, 57 all of 67 |
- harness: exit reasons {'completed': 69, 'step_limit': 3}; exit codes {'0': 69, '3': 3}; mean tool calls 4.57; failed tool calls 5; slowest item 27.319 s.
- harness_note: exit reasons {'completed': 71, 'step_limit': 1}; exit codes {'0': 71, '3': 1}; mean tool calls 4.15; failed tool calls 0; slowest item 26.276 s.
- harness_refined: exit reasons {'completed': 71, 'step_limit': 1}; exit codes {'0': 71, '3': 1}; mean tool calls 5.38; failed tool calls 1; slowest item 61.912 s.

## Accuracy by category (strict)

| Category | plain | rag | harness | harness_note | harness_refined |
|---|---|---|---|---|---|
| hard_chain | 0/9 | 0/9 | 8/9 | 9/9 | 7/9 |
| hard_distractor | 0/9 | 9/9 | 9/9 | 9/9 | 9/9 |
| hard_temporal | 0/9 | 7/9 | 8/9 | 9/9 | 9/9 |
| hard_filter | 0/9 | 0/9 | 9/9 | 9/9 | 8/9 |
| hard_compute | 0/9 | 4/9 | 5/9 | 7/9 | 9/9 |
| hard_partial | 4/9 | 7/9 | 8/9 | 9/9 | 9/9 |
| hard_counterfactual | 0/9 | 6/9 | 9/9 | 9/9 | 9/9 |
| hard_mixed | 0/9 | 9/9 | 9/9 | 9/9 | 7/9 |

## Grounded by category (KB-backed items: at least one / all supporting passages retrieved or read, of n)

| Category | plain | rag | harness | harness_note | harness_refined |
|---|---|---|---|---|---|
| hard_chain | n/a | 8 / 0 of 9 | 9 / 9 of 9 | 9 / 9 of 9 | 8 / 7 of 9 |
| hard_distractor | n/a | 9 / 9 of 9 | 9 / 9 of 9 | 9 / 9 of 9 | 9 / 9 of 9 |
| hard_temporal | n/a | 8 / 8 of 9 | 8 / 8 of 9 | 9 / 9 of 9 | 9 / 9 of 9 |
| hard_filter | n/a | 9 / 0 of 9 | 9 / 8 of 9 | 9 / 8 of 9 | 9 / 1 of 9 |
| hard_compute | n/a | 9 / 4 of 9 | 9 / 8 of 9 | 9 / 9 of 9 | 9 / 9 of 9 |
| hard_partial | n/a | 4 / 2 of 4 | 4 / 4 of 4 | 4 / 4 of 4 | 4 / 4 of 4 |
| hard_counterfactual | n/a | 9 / 6 of 9 | 9 / 9 of 9 | 9 / 9 of 9 | 9 / 9 of 9 |
| hard_mixed | n/a | 9 / 9 of 9 | 9 / 9 of 9 | 9 / 9 of 9 | 9 / 9 of 9 |

## Mean wall time per item by category (s, median in brackets)

| Category | plain | rag | harness | harness_note | harness_refined |
|---|---|---|---|---|---|
| hard_chain | 6.44 (5.7) | 4.24 (3.53) | 4.24 (4.01) | 5.63 (5.16) | 5.5 (4.06) |
| hard_distractor | 2.41 (0.76) | 1.25 (1.25) | 2.48 (2.23) | 2.07 (1.99) | 1.64 (1.48) |
| hard_temporal | 3.29 (3.14) | 5.13 (3.29) | 5.65 (4.65) | 5.42 (3.9) | 12.34 (4.89) |
| hard_filter | 4.1 (2.25) | 3.97 (3.8) | 17.49 (17.43) | 13.43 (16.0) | 21.09 (14.37) |
| hard_compute | 4.49 (3.7) | 4.67 (4.15) | 8.24 (6.35) | 8.11 (5.64) | 10.68 (8.81) |
| hard_partial | 3.71 (1.78) | 3.25 (2.8) | 5.54 (3.27) | 4.64 (4.38) | 6.27 (6.58) |
| hard_counterfactual | 5.93 (4.12) | 3.77 (2.75) | 3.49 (2.99) | 3.87 (3.98) | 4.07 (3.92) |
| hard_mixed | 5.43 (2.51) | 3.11 (3.3) | 3.05 (3.05) | 3.81 (3.05) | 7.2 (5.63) |

## Mean model calls per item by category

| Category | plain | rag | harness | harness_note | harness_refined |
|---|---|---|---|---|---|
| hard_chain | 1 | 1 | 5 | 5.44 | 7.11 |
| hard_distractor | 1 | 1 | 3.33 | 3 | 2.78 |
| hard_temporal | 1 | 1 | 4.11 | 4.33 | 4.33 |
| hard_filter | 1 | 1 | 7.22 | 6.56 | 10.67 |
| hard_compute | 1 | 1 | 9 | 7.78 | 9 |
| hard_partial | 1 | 1 | 6 | 5 | 6.44 |
| hard_counterfactual | 1 | 1 | 4.33 | 3.89 | 3.89 |
| hard_mixed | 1 | 1 | 3.44 | 4.33 | 6.44 |

## Mean prompt tokens per item by category

| Category | plain | rag | harness | harness_note | harness_refined |
|---|---|---|---|---|---|
| hard_chain | 129 | 298 | 38005 | 39244 | 16790 |
| hard_distractor | 121 | 293 | 16890 | 16162 | 5458 |
| hard_temporal | 130 | 329 | 23888 | 25926 | 12204 |
| hard_filter | 147 | 350 | 59427 | 48830 | 48293 |
| hard_compute | 138 | 307 | 61581 | 57894 | 34063 |
| hard_partial | 142 | 311 | 36621 | 30578 | 17301 |
| hard_counterfactual | 128 | 291 | 23102 | 19821 | 7524 |
| hard_mixed | 134 | 303 | 17792 | 26018 | 18588 |

