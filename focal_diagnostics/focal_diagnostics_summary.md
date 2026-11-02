# FOCAL_MAJORITY diagnostics: selection or noise?

Script: `focal_diagnostics.py`. Running `python focal_diagnostics.py [1] [2] [3] [p]` with no arguments runs every step (about 30 min; the placebo step uses 3 worker processes). All tables are CSVs in this folder. SMD = mean difference / sqrt((sd1² + sd0²)/2). The flag covers 57 of 1,644 firm clusters and 43 of 1,271 solo clusters, so read effect sizes and CIs, not stars.

## Model specs used
- **Firm (authoritative):** the paper tables A–F in `C:\Users\ensig\Claude Model Selection for Exploit\analysis\MODELS_FOR_PAPER.md`, built by `13_direct_effects.py` (A, B), `10_full_comparison.py` with no weights (C, D) and `11_theory_aligned.py` / `14_topic_size_robustness.py` (E, F).
  - DV: `FWDSELFCITES`.
  - Estimators: Poisson QMLE (HC1), Logit on 1[y>0] (HC1), ZT-Poisson (SE × √φ).
  - FE: SIC2 or SIC4. No clustering.
  - Hypothesis terms: H1 `IHSAGE2(_C2):BTDUMMY`, H2 `LOG5YRCT:BTDUMMY`, H3 `IHSSALES:BTDUMMY`.
  - **Every coefficient and SE in all 6 tables × 3 estimators reproduces to within 5e-5, at N = 6,516 (3,341 with y > 0)** (`step3_reproduction_check.csv`).
  - Caveat: the paper's ZT-Poisson BFGS fit stops at maxiter. A fully converged Newton fit has a log-likelihood about 50 units higher, but the hypothesis coefficients move by 0.006 or less. Both versions are reported.
- **Solo (PROVISIONAL):** no spec was found. Every `_est_e_*` marker equals 1 on every row, so the markers carry no information. I used `clogit isbt ~ lIExperience + lIFamiliarity + lIDiversity + logbackcites + focalpatentnumclass + isus + issmallnum`, grouped by btset. Please replace it with the real spec.

## Concentration of the flag (`flag_concentration_by_holder.csv`, `firm_flag_by_sic2.csv`)
- **Firm:** the 57 flagged clusters belong to 33 of 357 firms. DuPont has 7, Micron 7, Toshiba 5, Canon 3 and Ionis 3; the top 3 firms hold 33%. SIC 28 (chemicals and pharma) has 20 flagged clusters, a 14% flag rate against 2–3% elsewhere.
- **Solo:** the 43 flagged clusters are spread widely. Pioneer Hi-Bred has 5 and Silverbrook 4; the top 2 hold 21%.
- Dropping the top-2 holders' flagged clusters barely changes any SMD (column `smd_excl_top2_holders`), so no single firm drives the result.

## Step 1: largest differences at the breakthrough (BT) level (`step1_descriptives.csv`)
q-values are BH across the variable family, from firm-clustered (firm data) or holder-clustered (solo data) tests. Topic size itself differs by construction: log N has SMD −1.07 (firm) and −0.82 (solo).

| Firm, BT rows | SMD | q | Solo, BT rows | SMD | q |
|---|---|---|---|---|---|
| log1p FWDSELFCITES (DV) | +0.79 | 0.002 | appyear | +0.78 | – |
| KPST pctile 0-10y | −0.65 | 0.016 | KPST resid 0-10y | −0.66 | 0.004 |
| KPST resid 0-10y | −0.63 | 0.002 | iscorp (mechanical*) | +0.64 | <0.001 |
| KPST pctile 0-5y | −0.59 | 0.016 | KPST top10 0-10y | −0.60 | 0.08 |
| KPST resid 0-5y | −0.50 | 0.016 | newlIDiversity | +0.56 | 0.001 |
| KPST top10 0-10y | −0.45 | 0.24 | lIDiversity | +0.54 | 0.001 |
| log1p PRIORCITEUSAGE5YR | +0.44 | 0.018 | KPST resid 0-5y | −0.53 | 0.004 |
| APPYEAR | +0.41 | – | KPST pctile 0-10y | −0.49 | 0.11 |
| log1p NEWBACKCITES5YR | −0.32 | 0.17 | KPST pctile 0-5y | −0.47 | 0.05 |
| KPST top10 0-5y | −0.32 | 0.24 | KPST pctile 0-1y | −0.46 | 0.04 |

*In the solo data, clusters whose BT patent has no resolved holder (195 of them) can never be flagged. The `solo_assigned` rows repeat the solo analysis without them, and the conclusions hold.

The firm IVs are mostly balanced at BT level: IHSSALES, IHSAGE2 and LOG5YRCT all have |SMD| < 0.2. The comparison rows of flagged clusters show the same DV and Kelly pattern: FWDSELFCITES has SMD +0.55 and KPST resid 0-10y has SMD −0.83.

## Kelly measures (`step1_kelly_gap.csv`)
- Flagged breakthroughs score lower on KPST importance, not higher. For example, pctile 0-10y is 0.60 against 0.80.
- About half to two-thirds of that difference goes away once topic size and year are controlled. Adjusted pctile 0-10y: −0.11 [−0.24, 0.01].
- **The within-cluster gap (BT minus comparison mean) does not differ by flag in either dataset.** Firm pctile 0-10y gap diff: +0.01 [−0.04, 0.07]. The gap-family permutation p is 0.60 (firm) and 0.85 (solo).
- So the flag does not coincide with inflated textual distinctiveness of the breakthrough. Flagged clusters are simply less "important" in textual terms overall.

## Step 2: selection vs noise (`step2_*.csv`)
- **Stratified permutation** (2,000 draws, flag reassigned within deciles of topic size) over all 62 firm and 66 solo variables:
  - Firm: 23 variables with |t| > 1.96 against a null mean of 9.7 (p = 0.023). Mean |SMD| 0.28 against 0.15 (p = 0.001).
  - Solo: 31 against 7.3 (p = 0.001).
  - These are more differences than noise produces. The family of firm IVs/controls alone is borderline (5 against 1.9, p = 0.12). What drives the firm result is the DV, the Kelly levels and PRIORCITEUSAGE.
- **After conditioning** on topic size, year and (firm) IHSSALES, and in size- and year-matched nearest-neighbour comparisons:
  - The Kelly differences shrink to SMD −0.1 to −0.4 and are no longer significant (q > 0.3).
  - The **DV difference stays: adjusted SMD +0.84, q < 0.001.**
  - Solo IDiversity stays at about +0.45 (q ≈ 0.05).
- **Within firm** (27 firms with both flagged and unflagged clusters, firm FE): FWDSELFCITES SMD +0.71 (q = 0.001), PRIORCITEUSAGE +0.43, NEWBACKCITES −0.37. The Kelly differences disappear (SMD about −0.1 to −0.2, n.s.).

## Step 3: impact on the paper's results (`step3_paper_tables_full_vs_drop.csv`, `step3_placebo_summary.csv`)
The "drop" sample removes whole flagged clusters: N goes from 6,516 to 6,292 and clusters from 1,629 to 1,573. The placebo drops random unflagged clusters matched on topic-size decile (500 draws).

Headline Table F, full → drop (Δ in SE units; placebo z):

| Term | Poisson | Logit | ZT-Pois |
|---|---|---|---|
| BTDUMMY | 1.06 → 1.35 (+0.36 SE; z = 2.4) | 1.60 (p .039) → 1.47 (p .062) (−0.17; z = −1.2) | 1.27 → 1.60 (+0.76; z = 3.0) |
| IHSSALES:BT (H3) | −.054 → −.083 (−0.44; z = −2.4) | −.170 (p .031) → −.163 (p .040) (z = 0.4) | −.052 (p .23) → −.082 (p .074) (z = −2.6) |
| LOG5YRCT:BT (H2) | .043 → .061 (z = 2.0) | .138 (p .018) → .137 (p .021) (z = −0.1) | .017 → .031 (z = 1.7) |
| IHSAGE2:BT (H1) | −.071 → −.100 (z = −1.9) | stable | −.096 (p .14) → −.128 (p .059) (z = −2.2) |

- **Logit (participation), where H2 and H3 are significant, is stable in every table:** all placebo p-values are above 0.2.
- In the count and intensity models (Poisson, ZTP), dropping the flagged clusters moves coefficients 2–3 placebo SDs. It strengthens BTDUMMY and the H3/H1 moderators. No hypothesis coefficient changes sign.
- The controls LOG5YRCT, SLACK, X5YRDISPERSION and LOGFWDCITES shift by 3–8 placebo SDs.
- The same pattern holds at topic-size thresholds 10 and 30. At 50, with only 24 flagged clusters, the shifts shrink (`step3_threshold_models.csv`).
- **Interactions (step 3c):** all FOCAL_MAJORITY terms are jointly significant (p < 0.001), mostly through the lower-order FM terms. No hypothesis × FM triple interaction is significant, and the CIs are wide.
- **Solo (provisional clogit):** stable. Shifts are under 0.25 SE, placebo p is between 0.25 and 0.64, and the joint IV×FM test gives p = 0.17.

## Overall read
**The flag reflects selection, not noise, and the selection is into the DV.** When a firm holds the majority of its topic, its breakthrough and comparison patents collect far more self-citations, even within the same firm. That is close to mechanical, because the firm owns most of the follow-on patents in the topic.

The flag is spread across firms rather than driven by one, with some concentration in SIC 28. It is not a "spurious novelty" artefact: the Kelly BT-minus-comparison gap does not differ by flag, and the Kelly level differences are mostly topic size and year.

For the paper, the participation-logit results (H2 and H3) are robust to dropping these clusters. The Poisson and ZTP BT effects and the moderators get somewhat stronger without them. Reporting the drop-flagged version as a robustness column seems justified.
