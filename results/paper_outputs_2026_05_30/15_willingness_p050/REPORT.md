# Willingness-to-Wait 2D Sensitivity (Daganzo-LGB-Hybrid @ P=0.5)

Operating point: WAITING_PENALTY P = **0.5 €/parcel/day**
Postponement windows: [1, 2, 3]
Share grid: 11 points 0 → 100%

## Headline numbers (cost in k€, baseline = all-daily 1,977 k€)

| Window | 0% willing | 50% willing | 100% willing | Saving @100% | ⌀ Wait @100% |
|---|---|---|---|---|---|
| 1 day | 1,977 | 1,977 | 1,977 | 0.0% | 0.000d |
| 2 day | 1,977 | 1,829 | 1,681 | 15.0% | 0.197d |
| 3 day | 1,977 | 1,810 | 1,643 | 16.9% | 0.238d |

*Cost blend formula:* `cost(share, window) = (1-share) · cost_daily + share · cost_batched(window, P=0.5)`. The batched portion picks the schedule minimising `cost + P · pkts · wait_days`. Wait days only accrue on the batched portion.