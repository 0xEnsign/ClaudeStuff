# Test 1: Kelly et al. (2021) as a validity check, within sample (exploit paper, firm sample)

Script: `test1_within_sample.py` (run `python test1_within_sample.py [a] [b] [c]`, about 3 min). Data: `bt_firm_dataset_segments_20262104c_with_topic_size_kpst_focal.csv`, 1,644 sets (one breakthrough plus three within-firm matched controls). Kelly scores are for each row's own patent. Inference is clustered by firm (GVKEY), because sets are nested in firms. The primary window is 0–5y; 0–1y and 0–10y are robustness checks.

## Headline numbers

| | Result (0–5y unless noted) | File |
|---|---|---|
| Coverage | 1,605 of 1,644 sets (350 firms) have the BT and ≥1 control scored. 0–10y: 1,292 sets (grants ≤ 2002). 0–1y: 1,630 | `test1_coverage.csv` |
| **Convergent: importance** | BT 76.0 vs controls 74.5 percentile. Within-set Δ = **+1.50 pts (SE 0.47, p = .001)**, 0.06 SD. BT beats its own controls' mean in 60% of sets (Wilcoxon p < .001). 0–10y: +2.04 (p < .001). 0–1y: +0.88 (p = .07) | `test1_convergent.csv` |
| Convergent: novelty | log backward similarity (prior 5y) **−0.13 (−0.15 SD, p < .001)**. BTs are textually more distinct from prior art | same |
| Convergent: forward similarity | Also *lower* for BTs: 0–1y −0.12, 2–5y −0.11, 6–10y −0.10 log (all p < .001). The expected sign was positive | same |
| **Discriminant: ranking** | Kelly ranks the BT above a matched control in **53.2% of pairs [51.5, 54.9]** (chance 50%). It picks the BT out of its set 27.8% of the time vs 25.2% by chance | `test1_clogit_fit.csv` |
| Discriminant: explained share | Within-set clogit pseudo-R² = **0.003 [0.000, 0.007]**. Upper bound with all components, squares, percentiles and flags: **0.018** (0–5y), **0.021 [0.015, 0.035]** (0–10y; concordance 0.58, top-1 35% vs 27%) | same, `test1_clogit_coefs.csv` |
| Where Kelly variance sits | **76%** of Kelly percentile variance is between sets (51% net of a same-structure permutation null; ICC 0.68). Firm alone explains 38% (32% net). BT status explains **0.25%** of within-set variance | `test1_variance_decomp.csv` |
| **Outcome robustness** | Adding Kelly pctile 0–5y and Kelly × BT to Tables E and F (Poisson, logit, ZTP), on the identical N = 6,349: no hypothesis coefficient moves more than **0.12 baseline SE**. Logit H2 (.134, p = .025) and H3 (−.159, p = .048) are unchanged, as is BTDUMMY. Kelly × BT is n.s. everywhere (p > .75) | `test1_outcome_robustness.csv` |

## Read
- **Convergent evidence is real but small.** Breakthroughs score significantly higher on Kelly importance than their own matched controls, and the difference grows with the window (0–1y < 0–5y < 0–10y). It runs through the novelty side: lower similarity to prior art.
- **Kelly does not confirm the impact side.** Backward and forward similarity correlate 0.95–0.99, even within sets. They mostly measure one "vocabulary commonness" factor, and breakthroughs sit lower on all of it. Kelly's forward similarity sums similarity to *all* later patents, so a narrow trajectory of tens to hundreds of follow-on patents barely moves it. Do not claim Kelly validates trajectory launch.
- **Discriminant evidence is strong.** Kelly explains less than 1% of within-set breakthrough status, and at most about 2% with every component combined. The reason is that three-quarters of Kelly's variance lies between our firm × class × time matched sets, the same composition the matching holds fixed. This parallels the population result (year × subclass explains about 60–73%; `Kelly et al Replication/variance_decomp`).
- **Results do not depend on Kelly importance.**

## Caveats
- Shared method: both measures are text-based (our LDA uses abstracts; Kelly uses TF-BIDF on full text). That makes the weak agreement more telling for discriminant validity. It also means the convergent evidence is not fully independent.
- Similarity components are logged and residualised on KPST-population grant-year means. Forward windows follow KPST coverage rules (2–5y for grants ≤ 2007, 6–10y for ≤ 2002), because later windows are truncated.
- 0–10y drops breakthroughs granted after 2002 (352 sets). 0–5y drops those after 2007.
- Upper-bound concordance and top-1 use the in-sample fitted index, so they are optimistic. They are still small.
- For the binary top-10% flags, "share of sets BT higher" is distorted by ties; use the set-FE estimate (0–5y +0.9 pp, n.s.; 0–10y +2.9 pp, p < .001).
- Side finding: conditional on total forward citations, Kelly importance relates *negatively* to firm self-citations (Table F Poisson −0.52, p < .001; logit −0.26, p = .045). Patents that are textually "important" draw relatively more follow-on from outside the firm.

## Draft paper sentences
> As a validity check, breakthroughs score higher than their own matched controls on Kelly et al.'s (2021) text-based importance measure (+1.5 percentile points at five years, p = .001; +2.0 at ten years, p < .001), driven by lower textual similarity to prior art. Yet Kelly importance ranks the breakthrough above its matched control in only 53% of pairs and explains less than 1% of within-set variation in breakthrough status (about 2% when all of its components are combined), because three-quarters of its variance lies between our firm-, class- and time-matched sets. Controlling for it changes none of our estimates by more than 0.12 standard errors.

Shorter alternative:
> Breakthroughs are modestly but significantly more novel than their matched controls by Kelly et al.'s (2021) text-based measure (p = .001), but that measure explains less than 1% of within-set variation in breakthrough status, and controlling for it leaves our results unchanged.
