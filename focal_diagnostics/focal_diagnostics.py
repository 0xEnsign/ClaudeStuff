"""
focal_diagnostics.py -- is FOCAL_MAJORITY==1 selection or noise?

Reads (never modifies):
  ../bt_firm_dataset_segments_20262104c_with_topic_size_kpst_focal.csv   (cluster BTID, firm GVKEY)
  ../SoloInventorsExport_with_topic_size_kpst_focal.csv                  (cluster btset)
Writes CSVs into this folder (focal_diagnostics/).

Model specs
  FIRM (authoritative, from C:/Users/ensig/Claude Model Selection for Exploit/analysis/
        14_topic_size_robustness.py THEORY_SPEC == headline "M1" in REPORT_IP_INTENSITY.md /
        Table F in MODELS_FOR_PAPER.md):
        FWDSELFCITES ~ BTDUMMY + controls (+ quadratic centred terms) + X:BTDUMMY moderators
        + C(SICCODE) [SIC4 FE, recommended] ; Poisson QMLE (HC1) and Logit on 1[y>0] (HC1).
        Hypothesis terms: H1 IHSAGE2:BTDUMMY & IHSAGE2_C2:BTDUMMY, H2 LOG5YRCT:BTDUMMY,
        H3 IHSSALES:BTDUMMY.
  SOLO (PROVISIONAL -- no spec found; all _est_e_* markers are 1 for every row):
        conditional logit isbt ~ lIExperience + lIFamiliarity + lIDiversity
        + logbackcites + focalpatentnumclass + isus + issmallnum, groups = btset.
Run:  python focal_diagnostics.py [1] [2] [3] [p]   (no args = all steps; ~40-60 min, process pool)
"""
import os, sys, math, time, warnings
for _v in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')   # one BLAS thread per worker (MemoryError otherwise)
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.genmod.families import Poisson, Binomial
from statsmodels.discrete.conditional_models import ConditionalLogit
from statsmodels.stats.multitest import multipletests
import patsy
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)
FIRM_CSV = os.path.join(BASE, 'bt_firm_dataset_segments_20262104c_with_topic_size_kpst_focal.csv')
SOLO_CSV = os.path.join(BASE, 'SoloInventorsExport_with_topic_size_kpst_focal.csv')
N_PERM = 2000
N_PLACEBO = 500
SEED = 20260927
N_WORKERS = max(1, min(3, (os.cpu_count() or 2) - 2))

def out(name):
    return os.path.join(HERE, name)

# ---------------------------------------------------------------- data prep
def ihs(x):
    return np.log(x + np.sqrt(x * x + 1.0))

def prep_firm():
    d = pd.read_csv(FIRM_CSV, low_memory=False)
    # --- identical to the user's prep() in 14_topic_size_robustness.py ---
    d['ROA'] = d['EBITDA'] / d['ASSETS']
    d['SLACK'] = d['ASSETS'] / d['LIABILITIES']
    d['IHSSALES'] = ihs(d['SALES'] / 1.8)
    d['LOGFWDCITES'] = np.where(d['NUMFWDCITES'] > 0, np.log(d['NUMFWDCITES']), 0.0)
    d['ZEROFWDCITES'] = (d['NUMFWDCITES'] == 0).astype(int)
    d['LOG5YRCT'] = np.where(d['X5YRCOUNT'] > 0, np.log(d['X5YRCOUNT']), -1.0)
    d['ZERO5YRCT'] = (d['X5YRCOUNT'] == 0).astype(int)
    d['LOGBUSSEG'] = np.where(d['NBUSSEG'] > 0, np.log(d['NBUSSEG'].clip(lower=1)), 0.0)
    d['SIC2'] = (d['SICCODE'] // 100).astype(int)
    for v in ['IHSAGE2', 'IHSSALES', 'LOG5YRCT']:
        m = d[v].mean()
        d[v + '_C'] = d[v] - m
        d[v + '_C2'] = d[v + '_C'] ** 2
    # --- extra descriptors for the diagnostics only ---
    d['LOG1P_FWDSELFCITES'] = np.log1p(d['FWDSELFCITES'])
    d['LOG1P_PRIORCITEUSAGE5YR'] = np.log1p(d['PRIORCITEUSAGE5YR'])
    d['LOG1P_NEWBACKCITES5YR'] = np.log1p(d['NEWBACKCITES5YR'])
    d['LOG1P_X10YRCOUNT'] = np.log1p(d['X10YRCOUNT'])
    d['LOG1P_RDEXP'] = np.log1p(d['RDEXP'])
    d['IP_INTENSITY'] = d['LOG5YRCT'] - d['IHSSALES']
    d['CLUSTER'] = d['BTID']
    d['ISBT'] = d['BTDUMMY']
    d['GROUP2'] = d['GVKEY'].astype(str)          # firm cluster
    d['YEAR'] = d['APPYEAR']
    d['LOG_TOPIC_N'] = np.log(d['BT_TOPIC_N_PATENTS'])
    return d

def prep_solo():
    d = pd.read_csv(SOLO_CSV, low_memory=False)
    d['LOG1P_numtechclasses'] = np.log1p(d['numtechclasses'])
    d['daysxp_years'] = d['daysxp'] / 365.25   # can be negative
    d['CLUSTER'] = d['btset']
    d['ISBT'] = d['isbt']
    h = d['FOCAL_HOLDER_ID'].fillna('').astype(str)
    d['GROUP2'] = np.where(h == '', 'UNASSIGNED:' + d['btset'].astype(str), h)  # holder cluster
    d['YEAR'] = d['appyear']
    d['LOG_TOPIC_N'] = np.log(d['BT_TOPIC_N_PATENTS'])
    return d

KPST_VARS = ['KPST_imp_resid_0_1y', 'KPST_imp_resid_0_5y', 'KPST_imp_resid_0_10y',
             'KPST_imp_pctile_0_1y', 'KPST_imp_pctile_0_5y', 'KPST_imp_pctile_0_10y',
             'KPST_breakthrough_top10_0_5y', 'KPST_breakthrough_top10_0_10y',
             'KPST_backward_sim_5y']
FIRM_VARS = {  # var -> role
    'LOG1P_FWDSELFCITES': 'DV (log1p)',
    'IHSSALES': 'IV H3 (size)', 'IHSAGE2': 'IV H1 (age)', 'LOG5YRCT': 'IV H2 (IP stock)',
    'ZERO5YRCT': 'control', 'LOGBUSSEG': 'control', 'X5YRDISPERSION': 'control',
    'ROA': 'control', 'SLACK': 'control', 'LOGFWDCITES': 'control', 'ZEROFWDCITES': 'control',
    'IP_INTENSITY': 'descriptor', 'LOG1P_RDEXP': 'descriptor', 'NUMCLASSES': 'patent',
    'IHSBACKCITES': 'patent', 'IHSBACKCLASS': 'patent', 'LOG1P_PRIORCITEUSAGE5YR': 'patent',
    'LOG1P_NEWBACKCITES5YR': 'patent', 'LOG1P_X10YRCOUNT': 'patent', 'X10YRDISPERSION': 'patent',
    'APPYEAR': 'timing',
}
SOLO_VARS = {
    'lIExperience': 'IV (prov.)', 'lIFamiliarity': 'IV (prov.)', 'lIDiversity': 'IV (prov.)',
    'EffectiveDiversityExclusive': 'IV alt', 'IFocus': 'IV alt', 'CharlesDiversity': 'IV alt',
    'newlIFamiliarity': 'IV alt', 'newlIDiversity': 'IV alt', 'newlIExperience': 'IV alt',
    'logbackcites': 'control', 'logfwdcites': 'outcome-ish', 'focalpatentnumclass': 'control',
    'LOG1P_numtechclasses': 'control', 'daysxp_years': 'control', 'IsFirstPatent': 'control',
    'iscorp': 'control', 'isus': 'control', 'isforeign': 'control', 'issmallnum': 'control',
    'lnPatentBreadth': 'control', 'centralnovelty_z': 'novelty', 'extremenovelty_z10': 'novelty',
    'appyear': 'timing',
}
for k in KPST_VARS:
    FIRM_VARS[k] = 'Kelly'
    SOLO_VARS[k] = 'Kelly'

# ---------------------------------------------------------------- helpers
def smd(x1, x0):
    x1 = x1[~np.isnan(x1)]; x0 = x0[~np.isnan(x0)]
    if len(x1) < 2 or len(x0) < 2:
        return np.nan
    s = math.sqrt((x1.var(ddof=1) + x0.var(ddof=1)) / 2)
    return (x1.mean() - x0.mean()) / s if s > 0 else np.nan

def pooled_sd(x1, x0):
    x1 = x1[~np.isnan(x1)]; x0 = x0[~np.isnan(x0)]
    return math.sqrt((x1.var(ddof=1) + x0.var(ddof=1)) / 2)

def ols_flag(sub, var, groups=None, extra=''):
    """OLS var ~ FOCAL_MAJORITY (+extra); returns b, lo, hi, p. HC1 if groups None."""
    s = sub.dropna(subset=[var]).copy()
    f = f'Q("{var}") ~ FOCAL_MAJORITY' + extra
    s = s.dropna(subset=[c for c in ['LOG_TOPIC_N', 'YEAR', 'IHSSALES'] if c in extra])
    try:
        if groups is None:
            r = smf.ols(f, s).fit(cov_type='HC1')
        else:
            r = smf.ols(f, s).fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(s[groups])[0]})
        b = r.params['FOCAL_MAJORITY']; se = r.bse['FOCAL_MAJORITY']
        return b, b - 1.96 * se, b + 1.96 * se, r.pvalues['FOCAL_MAJORITY'], len(s)
    except Exception:
        return (np.nan,) * 4 + (len(s),)

def bh(p):
    p = np.asarray(p, float); q = np.full_like(p, np.nan)
    ok = ~np.isnan(p)
    if ok.sum():
        q[ok] = multipletests(p[ok], method='fdr_bh')[1]
    return q

def size_bins(clus, nb=10):
    """decile bins of log topic size across clusters (one row per cluster)."""
    return pd.qcut(clus['LOG_TOPIC_N'].rank(method='first'), nb, labels=False).values

# ---------------------------------------------------------------- STEP 1
def descriptives(d, varmap, level, dataset, clus2_label):
    sub = d[d['ISBT'] == (1 if level == 'BT' else 0)]
    rows = []
    top2 = (d[(d.ISBT == 1) & (d.FOCAL_MAJORITY == 1)].groupby('GROUP2').size()
            .sort_values(ascending=False).index[:2].tolist())
    for v, role in varmap.items():
        x1 = sub.loc[sub.FOCAL_MAJORITY == 1, v].astype(float).values
        x0 = sub.loc[sub.FOCAL_MAJORITY == 0, v].astype(float).values
        g1 = None if level == 'BT' else 'CLUSTER'
        b, lo, hi, p1, n = ols_flag(sub, v, g1)
        _, lo2, hi2, p2, _ = ols_flag(sub, v, 'GROUP2')
        s_ex = sub[~((sub.FOCAL_MAJORITY == 1) & sub.GROUP2.isin(top2))]
        rows.append(dict(dataset=dataset, level=level, var=v, role=role,
                         n_flag=int(np.sum(~np.isnan(x1))), n_unflag=int(np.sum(~np.isnan(x0))),
                         mean_flag=np.nanmean(x1), mean_unflag=np.nanmean(x0),
                         median_flag=np.nanmedian(x1), median_unflag=np.nanmedian(x0),
                         diff=b, ci_lo=lo, ci_hi=hi,
                         p_cluster=p1, ci_lo_grp2=lo2, ci_hi_grp2=hi2, p_grp2=p2,
                         smd=smd(x1, x0),
                         smd_excl_top2_holders=smd(s_ex.loc[s_ex.FOCAL_MAJORITY == 1, v].astype(float).values, x0)))
    t = pd.DataFrame(rows)
    fam = t['role'] != 'timing'
    t['q_cluster'] = np.nan; t['q_grp2'] = np.nan
    t.loc[fam, 'q_cluster'] = bh(t.loc[fam, 'p_cluster'])
    t.loc[fam, 'q_grp2'] = bh(t.loc[fam, 'p_grp2'])
    t['p_cluster_label'] = 'HC1 (1 row per cluster)' if level == 'BT' else 'clustered on cluster ID'
    t['p_grp2_label'] = f'clustered on {clus2_label}'
    return t

def concentration(d, dataset, namecol):
    bt = d[d.ISBT == 1]
    fl = bt[bt.FOCAL_MAJORITY == 1]
    g = fl.groupby('GROUP2').agg(n_flagged=('CLUSTER', 'size'),
                                 example_name=(namecol, 'first'),
                                 median_topic_n=('BT_TOPIC_N_PATENTS', 'median'))
    tot = bt.groupby('GROUP2').size().rename('n_clusters_total')
    g = g.join(tot).sort_values('n_flagged', ascending=False)
    g['share_of_flagged'] = g['n_flagged'] / len(fl)
    g['cum_share'] = g['share_of_flagged'].cumsum()
    g['flag_rate_within_holder'] = g['n_flagged'] / g['n_clusters_total']
    g.insert(0, 'dataset', dataset)
    return g.reset_index().rename(columns={'GROUP2': 'holder_or_firm'})

def kelly_gap(d, dataset):
    rows = []; gaps = {}
    for v in ['KPST_imp_pctile_0_10y', 'KPST_imp_pctile_0_5y', 'KPST_imp_resid_0_10y', 'KPST_imp_resid_0_5y']:
        bt = d[d.ISBT == 1].set_index('CLUSTER')[v]
        cm = d[d.ISBT == 0].groupby('CLUSTER')[v].mean()
        c = d[d.ISBT == 1].set_index('CLUSTER')[['FOCAL_MAJORITY', 'GROUP2', 'LOG_TOPIC_N', 'YEAR']].copy()
        c['gap'] = bt - cm
        c['bt'] = bt
        gaps[v] = c['gap']
        for what in ['bt', 'gap']:
            s = c.dropna(subset=[what]).rename(columns={what: 'Y'})
            x1 = s.loc[s.FOCAL_MAJORITY == 1, 'Y'].values; x0 = s.loc[s.FOCAL_MAJORITY == 0, 'Y'].values
            b, lo, hi, p, n = ols_flag(s, 'Y', None)
            _, lo2, hi2, p2, _ = ols_flag(s, 'Y', 'GROUP2')
            ba, loa, hia, pa, _ = ols_flag(s, 'Y', 'GROUP2', ' + LOG_TOPIC_N + C(YEAR)')
            rows.append(dict(dataset=dataset, measure=v,
                             quantity='BT value' if what == 'bt' else 'BT minus mean(comparison) within cluster',
                             n_flag=len(x1), n_unflag=len(x0), mean_flag=x1.mean(), mean_unflag=x0.mean(),
                             diff=b, ci_lo=lo, ci_hi=hi, p_hc1=p, ci_lo_grp2=lo2, ci_hi_grp2=hi2, p_grp2=p2,
                             smd=smd(x1, x0), adj_diff_size_year=ba, adj_ci_lo=loa, adj_ci_hi=hia, adj_p_grp2=pa))
    return pd.DataFrame(rows), pd.DataFrame(gaps)

# ---------------------------------------------------------------- STEP 2
def cluster_frame(d, varmap):
    """one row per cluster: BT-row values, cluster means of comparison rows, Kelly gaps."""
    bt = d[d.ISBT == 1].set_index('CLUSTER')
    cm = d[d.ISBT == 0].groupby('CLUSTER')[list(varmap)].mean()
    c = bt[['FOCAL_MAJORITY', 'GROUP2', 'LOG_TOPIC_N', 'YEAR'] + (['IHSSALES', 'GVKEY'] if 'GVKEY' in bt else [])].copy()
    for v in varmap:
        c['BT__' + v] = bt[v].astype(float)
        c['CMP__' + v] = cm[v]
    return c

def adjusted(c, varmap, dataset, firm):
    rows = []
    fl = c.FOCAL_MAJORITY.values == 1
    bins = size_bins(c.reset_index())
    # NN matching (k=3, with replacement) on standardized (log size, year)
    Z = c[['LOG_TOPIC_N', 'YEAR']].astype(float)
    Z = ((Z - Z.mean()) / Z.std()).values
    i1 = np.where(fl)[0]; i0 = np.where(~fl)[0]
    nn = []
    for i in i1:
        dist = ((Z[i0] - Z[i]) ** 2).sum(1)
        nn.append(i0[np.argsort(dist)[:3]])
    nn = np.concatenate(nn)
    extra = ' + LOG_TOPIC_N + C(YEAR)' + (' + IHSSALES' if firm else '')
    for v, role in varmap.items():
        col = 'BT__' + v
        x = c[col].values
        x1, x0 = x[fl], x[~fl]
        sd = pooled_sd(x1, x0)
        s = c.rename(columns={col: 'Y'})
        b, lo, hi, p, n = ols_flag(s, 'Y', 'GROUP2', extra)
        # stratified (ATT weights within size deciles)
        num = 0.0; wsum = 0.0
        for bb in np.unique(bins):
            m1 = (bins == bb) & fl & ~np.isnan(x); m0 = (bins == bb) & ~fl & ~np.isnan(x)
            if m1.sum() and m0.sum():
                num += m1.sum() * (x[m1].mean() - x[m0].mean()); wsum += m1.sum()
        strat = (num / wsum) / sd if wsum else np.nan
        xm = x[nn]
        nn_smd = (np.nanmean(x1) - np.nanmean(xm)) / sd
        row = dict(dataset=dataset, var=v, role=role, raw_smd=smd(x1, x0),
                   adj_diff=b, adj_ci_lo=lo, adj_ci_hi=hi, adj_p_grp2=p,
                   adj_smd=b / sd if sd else np.nan,
                   size_stratified_smd=strat, nn_size_year_matched_smd=nn_smd,
                   adj_controls='log topic N + appyear FE' + (' + IHSSALES' if firm else ''))
        if firm and v not in ('IHSSALES',):
            # within-firm (firm FE) among firms with both flagged and unflagged BTs
            g = c.groupby('GVKEY').FOCAL_MAJORITY.agg(['min', 'max'])
            both = g[(g['min'] == 0) & (g['max'] == 1)].index
            sw = s[s.GVKEY.isin(both)].dropna(subset=['Y'])
            try:
                r = smf.ols('Y ~ FOCAL_MAJORITY + C(GVKEY)', sw).fit(
                    cov_type='cluster', cov_kwds={'groups': pd.factorize(sw['GVKEY'])[0]})
                bw, sew = r.params['FOCAL_MAJORITY'], r.bse['FOCAL_MAJORITY']
                row.update(withinfirm_diff=bw, withinfirm_ci_lo=bw - 1.96 * sew,
                           withinfirm_ci_hi=bw + 1.96 * sew, withinfirm_p=r.pvalues['FOCAL_MAJORITY'],
                           withinfirm_smd=bw / sd, withinfirm_n_firms=len(both),
                           withinfirm_n_flag=int(sw.FOCAL_MAJORITY.sum()), withinfirm_n=len(sw))
            except Exception:
                pass
        rows.append(row)
    t = pd.DataFrame(rows)
    fam = t.role != 'timing'
    t['adj_q'] = np.nan
    t.loc[fam, 'adj_q'] = bh(t.loc[fam, 'adj_p_grp2'])
    if 'withinfirm_p' in t:
        t['withinfirm_q'] = np.nan
        t.loc[fam, 'withinfirm_q'] = bh(t.loc[fam, 'withinfirm_p'])
    return t

def permutation(c, cols, bins, n_perm, rng):
    """Stratified (within size bins) reassignment of the flag across clusters."""
    X = c[cols].values.astype(float)
    M = ~np.isnan(X); X0 = np.where(M, X, 0.0)
    g_obs = c.FOCAL_MAJORITY.values.astype(float)
    G = np.empty((n_perm + 1, len(g_obs)))
    G[0] = g_obs
    idx_by_bin = [np.where(bins == b)[0] for b in np.unique(bins)]
    for k in range(1, n_perm + 1):
        g = np.zeros_like(g_obs)
        for ib in idx_by_bin:
            n1 = int(g_obs[ib].sum())
            if n1:
                g[rng.choice(ib, n1, replace=False)] = 1
        G[k] = g
    H = 1 - G
    def stats_(W):
        n = W @ M; s = W @ X0; ss = W @ (X0 ** 2)
        m = s / n; v = (ss - n * m ** 2) / (n - 1)
        return n, m, v
    n1, m1, v1 = stats_(G); n0, m0, v0 = stats_(H)
    S = (m1 - m0) / np.sqrt((v1 + v0) / 2)
    T = (m1 - m0) / np.sqrt(v1 / n1 + v0 / n0)
    return S, T   # row 0 = observed

def perm_tables(c, varmap, dataset, rng):
    bins = size_bins(c.reset_index())
    rows_v = []; rows_g = []
    fams = {
        'BT rows: IVs/controls/patent': ['BT__' + v for v, r in varmap.items() if r not in ('timing', 'Kelly')],
        'BT rows: Kelly': ['BT__' + v for v, r in varmap.items() if r == 'Kelly'],
        'Comparison-row cluster means: IVs/controls/patent': ['CMP__' + v for v, r in varmap.items() if r not in ('timing', 'Kelly')],
        'Comparison-row cluster means: Kelly': ['CMP__' + v for v, r in varmap.items() if r == 'Kelly'],
        'Kelly within-cluster gap (BT - comparison mean)': [col for col in c.columns if col.startswith('GAP__')],
    }
    fams['ALL (BT + comparison + gaps)'] = sum(fams.values(), [])
    for fam, cols in fams.items():
        S, T = permutation(c, cols, bins, N_PERM, rng)
        aS = np.abs(S)
        obs, null = aS[0], aS[1:]
        if fam.startswith('ALL') is False:
            for j, col in enumerate(cols):
                rows_v.append(dict(dataset=dataset, family=fam, var=col, obs_smd=S[0, j],
                                   null_sd_of_smd=np.nanstd(S[1:, j]),
                                   null_95pct_abs_smd=np.nanpercentile(null[:, j], 95),
                                   perm_p_two_sided=(np.sum(null[:, j] >= obs[j]) + 1) / (N_PERM + 1)))
        for label, f in [('mean |SMD|', lambda a, t: np.nanmean(a, 1)),
                         ('max |SMD|', lambda a, t: np.nanmax(a, 1)),
                         ('# vars |SMD|>0.2', lambda a, t: np.sum(a > 0.2, 1)),
                         ('# vars |Welch t|>1.96', lambda a, t: np.sum(np.abs(t) > 1.96, 1))]:
            vals = f(aS, T)
            rows_g.append(dict(dataset=dataset, family=fam, n_vars=len(cols), statistic=label,
                               observed=vals[0], null_mean=vals[1:].mean(),
                               null_p95=np.percentile(vals[1:], 95),
                               perm_p=(np.sum(vals[1:] >= vals[0]) + 1) / (N_PERM + 1),
                               n_perm=N_PERM, strata='deciles of log BT_TOPIC_N_PATENTS'))
    tv = pd.DataFrame(rows_v)
    tv['perm_q_within_family'] = tv.groupby('family').perm_p_two_sided.transform(lambda p: bh(p.values))
    return tv, pd.DataFrame(rows_g)

# ---------------------------------------------------------------- STEP 3: firm models
# Paper tables = MODELS_FOR_PAPER.md Tables A-F (C:/Users/ensig/Claude Model Selection for Exploit/analysis),
# produced by 13_direct_effects.py (A,B), 10_full_comparison.py no-weights (C = D_SIC4_reduced,
# D = B_SIC4_full), 11_theory_aligned.py (E,F).  Estimators exactly as in those scripts:
#   Poisson QMLE GLM HC1; Logit GLM on 1[y>0] HC1 (same X); ZT-Poisson on y>0 via
#   statsmodels TruncatedLFPoisson (BFGS, maxiter 600, start = Poisson on y>0), SE x sqrt(phi).
DIRECT = ('BTDUMMY + IHSSALES + IHSSALES_C2 + IHSAGE2 + IHSAGE2_C2 + LOG5YRCT + LOG5YRCT_C2 '
          '+ ZERO5YRCT + LOGBUSSEG + X5YRDISPERSION + ROA + SLACK + LOGFWDCITES + ZEROFWDCITES')
REDUCED = (DIRECT + ' + IHSSALES:BTDUMMY + IHSAGE2:BTDUMMY + LOG5YRCT:BTDUMMY + ZERO5YRCT:BTDUMMY'
           ' + LOGBUSSEG:BTDUMMY + X5YRDISPERSION:BTDUMMY + ROA:BTDUMMY + SLACK:BTDUMMY')
FULLQ = REDUCED + ' + IHSSALES_C2:BTDUMMY + IHSAGE2_C2:BTDUMMY + LOG5YRCT_C2:BTDUMMY'
THEORY = (DIRECT + ' + IHSSALES:BTDUMMY + IHSAGE2:BTDUMMY + IHSAGE2_C2:BTDUMMY + LOG5YRCT:BTDUMMY'
          ' + ZERO5YRCT:BTDUMMY + LOGBUSSEG:BTDUMMY + X5YRDISPERSION:BTDUMMY + ROA:BTDUMMY + SLACK:BTDUMMY')
FIRM_RHS = THEORY
PAPER_TABLES = {  # label: (rhs, FE column)
    'A (SIC2, direct)': (DIRECT, 'SIC2'),
    'B (SIC4, direct)': (DIRECT, 'SICCODE'),
    'C (SIC4, x BT)': (REDUCED, 'SICCODE'),
    'D (SIC4, x BT + quad x BT)': (FULLQ, 'SICCODE'),
    'E (SIC2, preferred)': (THEORY, 'SIC2'),
    'F (SIC4, preferred/headline)': (THEORY, 'SICCODE'),
}
FIRM_NEEDED = ['FWDSELFCITES', 'BTDUMMY', 'IHSSALES', 'IHSAGE2', 'LOG5YRCT', 'ZERO5YRCT', 'X5YRDISPERSION',
               'ROA', 'SLACK', 'LOGFWDCITES', 'ZEROFWDCITES', 'LOGBUSSEG', 'NBUSSEG', 'SICCODE', 'SIC2', 'TOPIC_SIZE']
FM_INT = (' + FOCAL_MAJORITY + FOCAL_MAJORITY:BTDUMMY'
          ' + FOCAL_MAJORITY:IHSSALES + FOCAL_MAJORITY:IHSSALES:BTDUMMY'
          ' + FOCAL_MAJORITY:IHSAGE2 + FOCAL_MAJORITY:IHSAGE2:BTDUMMY'
          ' + FOCAL_MAJORITY:IHSAGE2_C2 + FOCAL_MAJORITY:IHSAGE2_C2:BTDUMMY'
          ' + FOCAL_MAJORITY:LOG5YRCT + FOCAL_MAJORITY:LOG5YRCT:BTDUMMY')
EST = ['Poisson QMLE (HC1)', 'Logit 1[y>0] (HC1)', 'ZT-Pois (BFGS as paper, SE x sqrt(phi))',
       'ZT-Pois (Newton, fully converged, SE x sqrt(phi))']

def canon(term):
    return ':'.join(sorted(term.split(':')))

def ztp_newton(y, X, b0, maxit=200):
    """Zero-truncated Poisson MLE by damped Newton with analytic derivatives (pinv for rank deficiency)."""
    b = np.asarray(b0, float).copy()
    def ll(b):
        eta = np.clip(X @ b, -50, 50); mu = np.exp(eta)
        return np.sum(y * eta - mu - np.log(-np.expm1(-mu)))
    L = ll(b)
    for it in range(maxit):
        eta = np.clip(X @ b, -50, 50); mu = np.exp(eta); q = -np.expm1(-mu); lam = mu / q
        g = X.T @ (y - lam)
        w = lam * (1 - mu * np.exp(-mu) / q)
        step = np.linalg.pinv((X * w[:, None]).T @ X) @ g
        t = 1.0
        while True:
            Ln = ll(b + t * step)
            if Ln >= L - 1e-12 or t < 1e-4:
                break
            t /= 2
        b = b + t * step
        done = abs(Ln - L) < 1e-9
        L = Ln
        if done:
            break
    eta = np.clip(X @ b, -50, 50); mu = np.exp(eta); q = -np.expm1(-mu); lam = mu / q
    w = lam * (1 - mu * np.exp(-mu) / q)
    V = np.linalg.pinv((X * w[:, None]).T @ X)
    return b, np.sqrt(np.clip(np.diag(V), 0, None)), L

def _keep(X):
    k = X.std(axis=0) > 1e-10
    k.iloc[0] = True
    return X.loc[:, k]

def fit_three(d, rhs, fe, paper_ztp=True, robust=True, starts=None):
    """returns {estimator: DataFrame(b, se, p)} over non-FE terms, plus n and n_pos."""
    from statsmodels.discrete.truncated_model import TruncatedLFPoisson
    from scipy import stats
    y, X = patsy.dmatrices(f'FWDSELFCITES ~ {rhs} + C({fe})', d, return_type='dataframe')
    X = _keep(X); yv = y.values.ravel()
    res = {}
    def glm(yy, fam, key):
        sp = None if starts is None else starts[key].reindex(X.columns).fillna(0).values
        m = sm.GLM(yy, X, family=fam)
        kw = dict(cov_type='HC1') if robust else {}
        # IRLS can hit NaN weights under quasi-separation (ZEROFWDCITES in the logit), which numpy's
        # lstsq reports as MemoryError; retry without the warm start, then give up (NaN) for that draw.
        for spx in ([sp, None] if sp is not None else [None]):
            try:
                r = m.fit(start_params=spx, **kw)
                if np.all(np.isfinite(r.params)):
                    return pd.DataFrame({'b': r.params, 'se': r.bse, 'p': r.pvalues})
            except (MemoryError, np.linalg.LinAlgError, ValueError):
                pass
        if robust:
            raise RuntimeError('GLM failed on a reported (non-placebo) fit')
        return pd.DataFrame({'b': np.nan, 'se': np.nan, 'p': np.nan}, index=X.columns)
    res[EST[0]] = glm(yv, Poisson(), EST[0])
    res[EST[1]] = glm((yv > 0).astype(int), Binomial(), EST[1])
    pos = yv > 0
    Xp = _keep(X.loc[pos]); yp = yv[pos]
    pp = sm.GLM(yp, Xp.values, family=Poisson()).fit()
    phi = float(pp.pearson_chi2 / pp.df_resid); sq = math.sqrt(phi)
    if paper_ztp:
        z = TruncatedLFPoisson(yp.astype(int), Xp.values, truncation=0).fit(
            start_params=pp.params, method='bfgs', maxiter=600, disp=0)
        se = np.asarray(z.bse) * sq
        res[EST[2]] = pd.DataFrame({'b': z.params, 'se': se,
                                    'p': 2 * (1 - stats.norm.cdf(np.abs(z.params / se)))}, index=Xp.columns)
    b0 = pp.params if starts is None else starts[EST[3]].reindex(Xp.columns).fillna(0).values
    bn, sen, _ = ztp_newton(yp, Xp.values, b0)
    sen = sen * sq
    with np.errstate(divide='ignore', invalid='ignore'):
        res[EST[3]] = pd.DataFrame({'b': bn, 'se': sen, 'p': 2 * (1 - stats.norm.cdf(np.abs(bn / sen)))},
                                   index=Xp.columns)
    for k in res:
        res[k] = res[k][~res[k].index.str.startswith('C(')]
        res[k].index = [canon(t) for t in res[k].index]
    return res, len(d), int(pos.sum())

def _task_fit(args):
    table, sample_label, d = args
    rhs, fe = PAPER_TABLES[table]
    res, n, npos = fit_three(d, rhs, fe)
    return table, sample_label, res, n, npos, d.CLUSTER.nunique(), int(d[d.ISBT == 1].FOCAL_MAJORITY.sum())

def parse_paper_tables():
    """read reported b/se from MODELS_FOR_PAPER.md for the reproduction check."""
    import re
    path = r'C:/Users/ensig/Claude Model Selection for Exploit/analysis/MODELS_FOR_PAPER.md'
    rows = []; tab = None
    ests = [EST[0], EST[1], EST[2]]
    for line in open(path, encoding='utf-8'):
        m = re.match(r'### Table ([A-F])', line)
        if m:
            tab = [k for k in PAPER_TABLES if k.startswith(m.group(1) + ' ')][0]
            continue
        m = re.match(r'\| `([^`]+)` \|(.*)', line)
        if m and tab:
            cells = [c.strip() for c in m.group(2).split('|')[:3]]
            for est, c in zip(ests, cells):
                mm = re.match(r'([+-]?[\d.]+) \(([\d.]+)\)', c)
                if mm:
                    rows.append(dict(table=tab, estimator=est, term=canon(m.group(1)),
                                     paper_b=float(mm.group(1)), paper_se=float(mm.group(2))))
    return pd.DataFrame(rows)

# ---- placebo workers (module-level for Windows spawn) ----
_G = {}
def _init_firm(d, starts):
    _G['d'] = d; _G['starts'] = starts

def _placebo_firm(drop_ids):
    d = _G['d']
    s = d[~d.CLUSTER.isin(drop_ids)]
    out_ = {}
    import gc
    for table, (rhs, fe) in PAPER_TABLES.items():
        res = None
        for _attempt in range(3):   # transient MemoryError seen under the Windows process pool
            try:
                res, _, _ = fit_three(s, rhs, fe, paper_ztp=False, robust=False, starts=_G['starts'][table])
                break
            except MemoryError:
                gc.collect()
        if res is None:
            continue   # this draw's table is left NaN
        for est, df_ in res.items():
            for t, b in df_['b'].items():
                out_[f'{table}|{est}|{t}'] = b
    return out_

SOLO_IV = ['lIExperience', 'lIFamiliarity', 'lIDiversity']
SOLO_CTRL = ['logbackcites', 'focalpatentnumclass', 'isus', 'issmallnum']

def fit_clogit(s, extra_cols=()):
    cols = SOLO_IV + SOLO_CTRL + list(extra_cols)
    s = s.dropna(subset=cols)
    m = ConditionalLogit(s['isbt'].values, s[cols].astype(float), groups=s['btset'].values)
    return m.fit(disp=0, method='bfgs', maxiter=500)

def _init_solo(s):
    _G['s'] = s

def _placebo_solo(drop_ids):
    s = _G['s']
    r = fit_clogit(s[~s.btset.isin(drop_ids)])
    return [r.params[v] for v in SOLO_IV]

def placebo_draws(clus, bins, n, rng):
    """each draw: drop, within each size bin, as many random UNFLAGGED clusters as flagged ones."""
    fl = clus.FOCAL_MAJORITY.values == 1
    ids = clus.index.values
    draws = []
    for _ in range(n):
        dr = []
        for b in np.unique(bins):
            k = int((fl & (bins == b)).sum())
            if k:
                pool = ids[(~fl) & (bins == b)]
                dr.extend(rng.choice(pool, k, replace=False))
        draws.append(np.array(dr))
    return draws

def shift_stats(sh_obs, sh):
    sh = sh[~np.isnan(sh)]
    m, s = sh.mean(), sh.std()
    return dict(shift_obs=sh_obs, placebo_shift_mean=m, placebo_shift_sd=s,
                placebo_shift_p2_5=np.percentile(sh, 2.5), placebo_shift_p97_5=np.percentile(sh, 97.5),
                shift_obs_z=(sh_obs - m) / s if s > 0 else np.nan,
                percentile_of_obs=100 * np.mean(sh <= sh_obs),
                p_two_sided=(np.sum(np.abs(sh - m) >= abs(sh_obs - m)) + 1) / (len(sh) + 1),
                n_placebo=len(sh))

# ================================================================ main
def main():
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    firm = prep_firm(); solo = prep_solo()
    # clusters whose BT patent has no resolved holder can never be flagged (FOCAL_SHARE=0)
    solo_a = solo[~solo.GROUP2.str.startswith('UNASSIGNED:')].copy()
    print(f'firm rows {len(firm)}  clusters {firm.CLUSTER.nunique()}  flagged {firm[firm.ISBT==1].FOCAL_MAJORITY.sum()}')
    print(f'solo rows {len(solo)}  clusters {solo.CLUSTER.nunique()}  flagged {solo[solo.ISBT==1].FOCAL_MAJORITY.sum()}')
    only = set(sys.argv[1:])   # optional: run subset of steps, e.g. "1 2 3 p"
    run = lambda k: (not only) or (k in only)

    if run('1'):
        conc = pd.concat([concentration(firm, 'firm', 'FIRMNAME'), concentration(solo, 'solo', 'assignee')])
        conc.to_csv(out('flag_concentration_by_holder.csv'), index=False)
        fb = firm[firm.ISBT == 1]
        sic = fb.groupby('SIC2').agg(n_clusters=('CLUSTER', 'size'), n_flagged=('FOCAL_MAJORITY', 'sum'))
        sic['flag_rate'] = sic.n_flagged / sic.n_clusters
        sic.sort_values('n_flagged', ascending=False).to_csv(out('firm_flag_by_sic2.csv'))
        desc = pd.concat([descriptives(firm, FIRM_VARS, 'BT', 'firm', 'GVKEY (firm)'),
                          descriptives(firm, FIRM_VARS, 'CMP', 'firm', 'GVKEY (firm)'),
                          descriptives(solo, SOLO_VARS, 'BT', 'solo', 'focal holder of cluster'),
                          descriptives(solo, SOLO_VARS, 'CMP', 'solo', 'focal holder of cluster'),
                          descriptives(solo_a, SOLO_VARS, 'BT', 'solo_assigned', 'focal holder of cluster'),
                          descriptives(solo_a, SOLO_VARS, 'CMP', 'solo_assigned', 'focal holder of cluster')])
        extra = []
        for ds, d in [('firm', firm), ('solo', solo), ('solo_assigned', solo_a)]:
            b = d[d.ISBT == 1]
            for v in ['LOG_TOPIC_N', 'BT_TOPIC_TOP1_SHARE']:
                x1 = b.loc[b.FOCAL_MAJORITY == 1, v].values; x0 = b.loc[b.FOCAL_MAJORITY == 0, v].values
                extra.append(dict(dataset=ds, level='BT', var=v, role='topic (mechanical)', n_flag=len(x1),
                                  n_unflag=len(x0), mean_flag=x1.mean(), mean_unflag=x0.mean(),
                                  median_flag=np.median(x1), median_unflag=np.median(x0), smd=smd(x1, x0)))
        desc = pd.concat([desc, pd.DataFrame(extra)])
        desc.to_csv(out('step1_descriptives.csv'), index=False)
        kg = pd.concat([kelly_gap(firm, 'firm')[0], kelly_gap(solo, 'solo')[0], kelly_gap(solo_a, 'solo_assigned')[0]])
        kg.to_csv(out('step1_kelly_gap.csv'), index=False)
        print(f'step 1 done {time.time()-t0:.0f}s')

    if run('2'):
        adj_all = []; pv_all = []; pg_all = []
        for ds, d, vm, isfirm in [('firm', firm, FIRM_VARS, True), ('solo', solo, SOLO_VARS, False),
                                  ('solo_assigned', solo_a, SOLO_VARS, False)]:
            c = cluster_frame(d, vm)
            _, gaps = kelly_gap(d, ds)
            for v in gaps:
                c['GAP__' + v] = gaps[v]
            adj_all.append(adjusted(c, vm, ds, isfirm))
            tv, tg = perm_tables(c, vm, ds, rng)
            pv_all.append(tv); pg_all.append(tg)
        pd.concat(adj_all).to_csv(out('step2_adjusted_matched_withinfirm.csv'), index=False)
        pd.concat(pv_all).to_csv(out('step2_permutation_by_variable.csv'), index=False)
        pd.concat(pg_all).to_csv(out('step2_permutation_global.csv'), index=False)
        print(f'step 2 done {time.time()-t0:.0f}s')

    dfm = firm.dropna(subset=FIRM_NEEDED).copy()
    if run('3'):
        # ---- all paper tables, full vs drop-flagged, plus 10/30/50 thresholds
        samples = {'full': dfm, 'drop FOCAL_MAJORITY clusters': dfm[dfm.FOCAL_MAJORITY == 0]}
        for T in [10, 30, 50]:
            sT = dfm[dfm.BT_TOPIC_N_PATENTS >= T]
            samples[f'N>={T} full'] = sT
            samples[f'N>={T} drop FOCAL_MAJORITY clusters'] = sT[sT.FOCAL_MAJORITY == 0]
        tasks = [(tab, lab, s) for tab in PAPER_TABLES for lab, s in samples.items()]
        with ProcessPoolExecutor(N_WORKERS) as ex:
            results = list(ex.map(_task_fit, tasks))
        long = []
        for tab, lab, res, n, npos, ncl, nfl in results:
            for est, df_ in res.items():
                for t, r in df_.iterrows():
                    long.append(dict(table=tab, sample=lab, estimator=est, term=t, b=r.b, se=r.se, p=r.p,
                                     n=(npos if est.startswith('ZT') else n), n_clusters=ncl, n_flagged_clusters=nfl))
        long = pd.DataFrame(long)
        long.to_csv(out('step3_paper_tables_all_samples_long.csv'), index=False)
        # reproduction check
        pap = parse_paper_tables()
        full = long[long['sample'] == 'full']
        rep = pap.merge(full, on=['table', 'estimator', 'term'], how='left')
        rep['abs_diff_b'] = (rep.b - rep.paper_b).abs(); rep['abs_diff_se'] = (rep.se - rep.paper_se).abs()
        rep['reproduces'] = (rep.abs_diff_b <= 0.00015) & (rep.abs_diff_se <= 0.00015)
        rep.to_csv(out('step3_reproduction_check.csv'), index=False)
        print('reproduction: ', rep.groupby(['table', 'estimator']).reproduces.mean().to_string())
        # side-by-side tables
        sbs = []
        pairs = [('full', 'drop FOCAL_MAJORITY clusters')] + \
                [(f'N>={T} full', f'N>={T} drop FOCAL_MAJORITY clusters') for T in [10, 30, 50]]
        for a, b in pairs:
            A = long[long['sample'] == a].set_index(['table', 'estimator', 'term'])
            B = long[long['sample'] == b].set_index(['table', 'estimator', 'term'])
            j = A.join(B, lsuffix='_full', rsuffix='_drop', how='outer')
            j['comparison'] = f'{a}  vs  {b}'
            j['delta_b'] = j.b_drop - j.b_full
            j['delta_in_se_full'] = j.delta_b / j.se_full
            j['sig05_full'] = j.p_full < 0.05; j['sig05_drop'] = j.p_drop < 0.05
            j['sig10_full'] = j.p_full < 0.10; j['sig10_drop'] = j.p_drop < 0.10
            j['sig05_changes'] = j.sig05_full != j.sig05_drop
            j['sign_flips'] = np.sign(j.b_full) != np.sign(j.b_drop)
            sbs.append(j.reset_index())
        sbs = pd.concat(sbs)
        sbs.to_csv(out('step3_paper_tables_full_vs_drop.csv'), index=False)
        # (c) FM interactions on preferred tables E/F
        wald = []; coef = []
        for tab in ['E (SIC2, preferred)', 'F (SIC4, preferred/headline)']:
            rhs, fe = PAPER_TABLES[tab]
            y, X = patsy.dmatrices(f'FWDSELFCITES ~ {rhs}{FM_INT} + C({fe})', dfm, return_type='dataframe')
            X = _keep(X); yv = y.values.ravel()
            for est, yy, fam in [(EST[0], yv, Poisson()), (EST[1], (yv > 0).astype(int), Binomial())]:
                r = sm.GLM(yy, X, family=fam).fit(cov_type='HC1')
                idx = [k for k in r.params.index if 'FOCAL_MAJORITY' in k]
                for k in idx:
                    coef.append(dict(table=tab, estimator=est, term=canon(k), b=r.params[k], se=r.bse[k],
                                     ci_lo=r.params[k] - 1.96 * r.bse[k], ci_hi=r.params[k] + 1.96 * r.bse[k],
                                     p=r.pvalues[k], n=len(dfm)))
                bb = r.params[idx].values; V = r.cov_params().loc[idx, idx].values
                from scipy import stats
                W = float(bb @ np.linalg.pinv(V) @ bb)
                wald.append(dict(dataset='firm', table=tab, estimator=est, test='all FOCAL_MAJORITY terms = 0',
                                 W=W, df=len(idx), p=1 - stats.chi2.cdf(W, len(idx))))
        pd.DataFrame(coef).to_csv(out('step3c_firm_fm_interactions.csv'), index=False)
        # ---- solo (provisional clogit)
        scoef = []
        ss = solo.copy()
        for v in SOLO_IV:
            ss[v + ':FM'] = ss[v] * ss['FOCAL_MAJORITY']
        solo_samples = {'full': (solo, ()), 'drop FOCAL_MAJORITY clusters': (solo[solo.FOCAL_MAJORITY == 0], ()),
                        '(c) full + IVxFM': (ss, [v + ':FM' for v in SOLO_IV])}
        for T in [10, 30, 50]:
            sT = solo[solo.BT_TOPIC_N_PATENTS >= T]
            solo_samples[f'N>={T} full'] = (sT, ())
            solo_samples[f'N>={T} drop FOCAL_MAJORITY clusters'] = (sT[sT.FOCAL_MAJORITY == 0], ())
        for lab, (s, ex_) in solo_samples.items():
            r = fit_clogit(s, ex_)
            for t in r.params.index:
                scoef.append(dict(dataset='solo', estimator='clogit by btset (PROVISIONAL spec)', sample=lab,
                                  term=t, b=r.params[t], se=r.bse[t], p=r.pvalues[t], n=len(s),
                                  n_clusters=s.btset.nunique(), n_flagged_clusters=int(s[s.isbt == 1].FOCAL_MAJORITY.sum())))
            if ex_:
                idx = [i for i, t in enumerate(r.params.index) if t.endswith(':FM')]
                bvec = np.asarray(r.params)[idx]; V = np.asarray(r.cov_params())[np.ix_(idx, idx)]
                from scipy import stats
                W = float(bvec @ np.linalg.pinv(V) @ bvec)
                wald.append(dict(dataset='solo', table='provisional clogit', estimator='clogit',
                                 test='all IVxFM = 0', W=W, df=len(idx), p=1 - stats.chi2.cdf(W, len(idx))))
        scoef = pd.DataFrame(scoef)
        scoef.to_csv(out('step3_solo_clogit_provisional_long.csv'), index=False)
        ssb = []
        for a, b in [('full', 'drop FOCAL_MAJORITY clusters')] + \
                    [(f'N>={T} full', f'N>={T} drop FOCAL_MAJORITY clusters') for T in [10, 30, 50]]:
            A = scoef[scoef['sample'] == a].set_index('term'); B = scoef[scoef['sample'] == b].set_index('term')
            j = A[['b', 'se', 'p', 'n', 'n_clusters', 'n_flagged_clusters']].join(
                B[['b', 'se', 'p', 'n', 'n_clusters', 'n_flagged_clusters']], lsuffix='_full', rsuffix='_drop')
            j['comparison'] = f'{a}  vs  {b}'
            j['delta_b'] = j.b_drop - j.b_full; j['delta_in_se_full'] = j.delta_b / j.se_full
            j['sig05_full'] = j.p_full < 0.05; j['sig05_drop'] = j.p_drop < 0.05
            ssb.append(j.reset_index())
        pd.concat(ssb).to_csv(out('step3_solo_full_vs_drop.csv'), index=False)
        pd.DataFrame(wald).to_csv(out('step3c_fm_interaction_joint_wald.csv'), index=False)
        print(f'step 3 models done {time.time()-t0:.0f}s')

    if run('p'):
        long = pd.read_csv(out('step3_paper_tables_all_samples_long.csv'))
        psum = []
        # firm: all six tables x (Poisson, Logit, ZTP-Newton); non-robust warm-started fits
        cf = dfm[dfm.ISBT == 1].drop_duplicates('CLUSTER').set_index('CLUSTER')
        bins_f = size_bins(cf.reset_index())
        draws = placebo_draws(cf, bins_f, N_PLACEBO, rng)
        starts = {}
        for tab, (rhs, fe) in PAPER_TABLES.items():
            y, X = patsy.dmatrices(f'FWDSELFCITES ~ {rhs} + C({fe})', dfm, return_type='dataframe')
            X = _keep(X); yv = y.values.ravel()
            st = {EST[0]: sm.GLM(yv, X, family=Poisson()).fit().params,
                  EST[1]: sm.GLM((yv > 0).astype(int), X, family=Binomial()).fit().params}
            Xp = _keep(X.loc[yv > 0]); pp = sm.GLM(yv[yv > 0], Xp, family=Poisson()).fit()
            st[EST[3]] = pd.Series(ztp_newton(yv[yv > 0], Xp.values, pp.params.values)[0], index=Xp.columns)
            starts[tab] = st
        with ProcessPoolExecutor(N_WORKERS, initializer=_init_firm, initargs=(dfm, starts)) as ex:
            P = pd.DataFrame(list(ex.map(_placebo_firm, draws, chunksize=2)))
        P.to_csv(out('step3_placebo_draws_firm.csv'), index=False)
        for col in P.columns:
            tab, est, t = col.split('|')
            sel = lambda s_: long[(long['sample'] == s_) & (long.table == tab) & (long.estimator == est) & (long.term == t)]
            a, b = sel('full'), sel('drop FOCAL_MAJORITY clusters')
            if len(a) and len(b):
                row = dict(dataset='firm', table=tab, estimator=est, term=t,
                           b_full=a.b.iloc[0], se_full=a.se.iloc[0], b_drop=b.b.iloc[0])
                row.update(shift_stats(b.b.iloc[0] - a.b.iloc[0], P[col].values - a.b.iloc[0]))
                psum.append(row)
        print(f'firm placebo done {time.time()-t0:.0f}s')
        # solo
        cs = solo[solo.ISBT == 1].set_index('CLUSTER')
        bins_s = size_bins(cs.reset_index())
        draws = placebo_draws(cs, bins_s, N_PLACEBO, rng)
        with ProcessPoolExecutor(N_WORKERS, initializer=_init_solo, initargs=(solo,)) as ex:
            Ps = np.array(list(ex.map(_placebo_solo, draws, chunksize=8)))
        pd.DataFrame(Ps, columns=SOLO_IV).to_csv(out('step3_placebo_draws_solo.csv'), index=False)
        rf, re_ = fit_clogit(solo), fit_clogit(solo[solo.FOCAL_MAJORITY == 0])
        for j, t in enumerate(SOLO_IV):
            row = dict(dataset='solo', table='provisional clogit', estimator='clogit', term=t,
                       b_full=rf.params[t], se_full=rf.bse[t], b_drop=re_.params[t])
            row.update(shift_stats(re_.params[t] - rf.params[t], Ps[:, j] - rf.params[t]))
            psum.append(row)
        pd.DataFrame(psum).to_csv(out('step3_placebo_summary.csv'), index=False)
        print(f'all done {time.time()-t0:.0f}s')

if __name__ == '__main__':
    main()
