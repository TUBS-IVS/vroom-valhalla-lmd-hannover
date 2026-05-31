# 3D Sensitivity (window × share × service-penalty) — Daganzo-LGB-Hybrid

Baseline (all-daily): **1,977 k€**
Grid: 3 windows × 11 shares × 8 penalties = 264 cells

## Cost saving at maximum willingness (share = 100%)
| Window | $P=0$ | $P=0.25$ | $P=0.50$ | $P=1.0$ | $P=5.0$ |
|---|---|---|---|---|---|
| 1 day | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| 2 day | 19.4% | 17.8% | 15.0% | 10.6% | 0.0% |
| 3 day | 25.5% | 21.6% | 16.9% | 11.0% | 0.0% |

*Reading the cube:* Increasing **share** moves the customer-side (more accept batching). Increasing **window** loosens the structural constraint (more valid schedules). Increasing **penalty $P$** raises the operator-side cost of service, pushing the optimizer toward denser delivery. Saving is greatest at high share + wide window + low penalty; service-cost trade-off elbow remains near $P=0.5$.