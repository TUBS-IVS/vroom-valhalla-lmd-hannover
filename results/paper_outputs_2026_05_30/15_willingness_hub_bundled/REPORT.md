# Willingness × Hub-Bundled Express  — Daganzo-Hybrid v2-aug

**Operating point:** $P = 0.5$ €/parcel/day
**B2C express share:** `fast_share_b2c = s` (slider), **B2B express share:** `fast_share_b2b = s / 2` (mirrors abstract 10%/5%)
**Standalone threshold:** if express parcels ≥ 150 → standalone ML prediction
**Bundle target:** 1000 pkts, max 2000, area ≤ 100 km²

## Headline (window = 3 days, all 7 LSPs)

| Share willing | f_s (B2C) | Total cost [k€] | Batched | Standalone Express | Bundled Express | Saving vs s=100% |
|---|---|---|---|---|---|---|
| 100% | 0.00 | 1,626 | 1,626 | 0 | 0 | +13.64% |
| 90% | 0.10 | 1,820 | 1,752 | 1 | 67 | +3.34% |
| 80% | 0.20 | 1,816 | 1,739 | 9 | 69 | +3.53% |
| 70% | 0.30 | 1,814 | 1,725 | 26 | 63 | +3.64% |
| 60% | 0.40 | 1,823 | 1,730 | 39 | 54 | +3.19% |
| 50% | 0.50 | 1,834 | 1,747 | 41 | 47 | +2.56% |
| 40% | 0.60 | 1,839 | 1,739 | 69 | 31 | +2.30% |
| 30% | 0.70 | 1,850 | 1,756 | 73 | 22 | +1.72% |
| 20% | 0.80 | 1,859 | 1,772 | 69 | 18 | +1.23% |
| 10% | 0.90 | 1,871 | 1,801 | 57 | 13 | +0.64% |
| 0% | 1.00 | 1,883 | 1,839 | 35 | 8 | +0.00% |