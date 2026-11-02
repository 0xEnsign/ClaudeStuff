# Test 2: sample selection vs the patent population (Kelly et al. 2021 importance)

Exploit paper, firm sample only (1,644 breakthroughs, 4,932 controls). Script: `test2_selection.py` (seed 20261029, about 1 min). Results: `test2_benchmark.csv`, `test2_coverage.csv`, `test2_population_baseline*.csv`, `test2_checks.csv`, log `test2_run.log`.

## Method
- **Population:** all KPST-scored utility patents granted 1989–2010, 3.05M patents. Primary USPC class and subclass come from the Anna export (`variance_decomp/data/patents.parquet`, utility only, `pid_num` = patent number). All 6,576 sample patents are excluded from the pool.
- **Stratified null:** each sample patent is matched to a random pool patent from its own stratum. This is repeated over 1,000 replicate samples, separately for breakthroughs and controls. The three stratum tiers are:
  - grant year;
  - grant year × primary class (primary benchmark);
  - grant year × primary subclass.
- **Statistics:** mean percentile, share in Kelly's top 10% (pctile ≥ 0.90, which matches KPST's flags exactly), and median percentile. The p-value is a two-sided permutation p. KS is a one-sample test against the exact stratum-mixture population CDF.
- **Windows:** 0–5y is primary; 0–1y and 0–10y are robustness checks.

## Coverage and checks
- 6,514 of 6,576 sample patents are in the KPST universe, and all 6,514 have a USPC primary class and subclass.
- File and parquet percentiles agree (max difference 3e-8).
- 0–5y scores exist for 1,607 breakthroughs and 4,801 controls.
- Every sample patent has year and year × class peers.
- At the subclass tier, 187 breakthroughs and 501 controls have no non-sample peer in the same grant year × subclass, and the median stratum has only 7 pool patents.

## Baseline: the pooled percentile is not 0.5 / 10% in our years
Population grants in 1989–2007 (0–5y window) average a **0.48** percentile, with **13.9%** in the top 10%. For 0–10y (1989–2002) the figures are 0.47 and 16.5%. The benchmark must therefore be empirical.

## Results, 0–5y window (mean percentile; top-10% share)

| Benchmark | Breakthroughs | Matched random | Gap (pts / pop. SD) | Controls | Matched random | Gap (pts / pop. SD) |
|---|---|---|---|---|---|---|
| Grant year | 0.760; 43.0% | 0.477; 14.3% | +28.4 / 0.87 | 0.745; 42.0% | 0.478; 14.2% | +26.7 / 0.82 |
| Year × class | 0.760; 43.0% | 0.678; 32.9% | **+8.2 / 0.28** | 0.745; 42.0% | 0.673; 32.0% | **+7.2 / 0.24** |
| Year × subclass | 0.764; 43.2% | 0.718; 37.2% | +4.6 / 0.17 | 0.753; 42.7% | 0.715; 36.7% | +3.9 / 0.14 |

- Every gap is p < 0.002, the floor for 1,000 draws. Every KS test is p < 1e-6.
- **Common subset** (the same patents in all three tiers): breakthrough gaps are +28.7 → +8.1 → +4.6 points, and control gaps +27.6 → +7.4 → +3.9.
  - Technology class absorbs about 72% of the elevation relative to same-year patents.
  - Subclass absorbs about 84–86%.
- 0–1y and 0–10y show the same pattern. Year-only gaps are +23 and +31 points, year × class gaps +7 and +9, and year × subclass gaps +4 and +5.
- Breakthroughs sit only about 1 point above controls at every tier.

## Caveats
1. The population is all utility patents, including individuals, universities and foreign firms. It is not a public-firm benchmark, so part of the residual gap may simply reflect public-firm patents.
2. Kelly forward similarity measures the technology neighbourhood. Our breakthrough topics were selected to have at least 10 follow-on patents, which means growing areas. Some elevation at the class tier is therefore built in, and controls inherit it through subclass matching.
3. With thousands of patents, any selection is "significant" against a random-sampling null, so read the effect sizes. Small subclass strata make that tier's null noisier, but the common-subset rerun gives the same decomposition.
4. Grant year comes from KPST `issue_year`. The primary class assumes the first-listed USPC class is the original.

## Draft paper text
"Relative to random patents granted in the same year and primary technology class, both our breakthroughs and their matched controls score higher on Kelly et al.'s (2021) text-based importance measure (mean percentile 0.76 and 0.74 vs. 0.68 and 0.67; p < .01, 1,000 stratified draws). Most of their elevation over same-year patents (about 72%) reflects the technology classes in which they sit, and matching on grant year and subclass shrinks the remaining gap to 4–5 percentile points (0.14–0.17 SD). Breakthroughs and controls are nearly indistinguishable at every level of the comparison: our sample is drawn from active technology areas, and the within-firm matched design compares like with like."
