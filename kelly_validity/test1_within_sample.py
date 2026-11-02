"""Test 1: within-sample convergent and discriminant validity of the LDA breakthrough classification
against Kelly, Papanikolaou, Seru & Taddy (2021) importance. Exploit (firm) sample only.

Run:  python test1_within_sample.py [a] [b] [c]   (no args = all; writes test1_*.csv next to this file)
"""
import os
import sys
import math
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

warnings.filterwarnings('ignore')

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)
FIRM_CSV = os.path.join(BASE, 'bt_firm_dataset_segments_20262104c_with_topic_size_kpst_focal.csv')
KPST_POP = (r'C:\Users\ensig\Kelly et al Replication\Measuring-Technological-Innovation-Over-the-Long-Run-'
            r'Replication-Kit-master\novelty_scores\patent_novelty_scores.parquet')
FD_DIR = os.path.join(BASE, 'focal_diagnostics')
SEED = 20261029
N_BOOT = 1000
N_PERM = 500
# KPST window coverage rules (KPST_columns_README.txt): 0-5y through grant 2007, 0-10y through 2002
LAST_GRANT_5Y, LAST_GRANT_10Y = 2007, 2002


def out(name):
    return os.path.join(HERE, name)


# ---------------------------------------------------------------- data
def load():
    d = pd.read_csv(FIRM_CSV, low_memory=False)
    pop = pd.read_parquet(KPST_POP, columns=['issue_year', 'backward_sim_5y', 'forward_sim_0_1y',
                                             'forward_sim_2_5y', 'forward_sim_6_10y'])
    pop = pop[(pop.issue_year >= 1985) & (pop.issue_year <= 2010)]
    comps = {'R_LBWD': 'backward_sim_5y', 'R_LF01': 'forward_sim_0_1y',
             'R_LF25': 'forward_sim_2_5y', 'R_LF610': 'forward_sim_6_10y'}
    yr = d['KPST_issue_year']
    for new, src in comps.items():
        s = pop[src].where(pop[src] > 0)
        ymean = np.log(s).groupby(pop.issue_year).mean()
        x = d['KPST_' + src].where(d['KPST_' + src] > 0)
        d[new] = np.log(x) - yr.map(ymean)
    d.loc[yr > LAST_GRANT_5Y, 'R_LF25'] = np.nan
    d.loc[yr > LAST_GRANT_10Y, 'R_LF610'] = np.nan
    for w in ['0_1y', '0_5y', '0_10y']:
        d[f'PCT_{w}'] = 100 * d[f'KPST_imp_pctile_{w}']
    for w in ['0_5y', '0_10y']:
        d[f'TOP10_{w}'] = 100 * d[f'KPST_breakthrough_top10_{w}']
    return d


LABELS = {
    'PCT_0_5y': 'Importance percentile, 0-5y (primary)',
    'PCT_0_1y': 'Importance percentile, 0-1y',
    'PCT_0_10y': 'Importance percentile, 0-10y',
    'KPST_imp_resid_0_5y': 'Importance (log, year-residualised), 0-5y',
    'KPST_imp_resid_0_1y': 'Importance (log, year-residualised), 0-1y',
    'KPST_imp_resid_0_10y': 'Importance (log, year-residualised), 0-10y',
    'TOP10_0_5y': 'KPST top-10% breakthrough, 0-5y (pp)',
    'TOP10_0_10y': 'KPST top-10% breakthrough, 0-10y (pp)',
    'R_LBWD': 'log backward similarity 5y (novelty; lower = more novel)',
    'R_LF01': 'log forward similarity 0-1y',
    'R_LF25': 'log forward similarity 2-5y (grants <= 2007)',
    'R_LF610': 'log forward similarity 6-10y (grants <= 2002)',
}


def eligible(d, cols):
    """rows with all cols present, in sets where the BT and >= 1 control are present"""
    s = d.dropna(subset=cols)
    g = s.groupby('BTID')['BTDUMMY'].agg(['sum', 'size'])
    keep = g.index[(g['sum'] == 1) & (g['size'] >= 2)]
    return s[s.BTID.isin(keep)].copy()


# ---------------------------------------------------------------- (a) convergent
def coverage(d):
    rows = []
    for v in LABELS:
        s = eligible(d, [v])
        sz = s.groupby('BTID').size()
        rows.append(dict(variable=v, label=LABELS[v], rows_scored=int(d[v].notna().sum()),
                         sets_eligible=len(sz), sets_complete=int((sz == 4).sum()),
                         rows_eligible=len(s), firms=s.GVKEY.nunique()))
    return pd.DataFrame(rows)


def fe_regression(s, v):
    """v ~ BTDUMMY with set FE (within transform), SE clustered by firm (sets nest in firms)"""
    y = s[v] - s.groupby('BTID')[v].transform('mean')
    x = s['BTDUMMY'] - s.groupby('BTID')['BTDUMMY'].transform('mean')
    r = sm.OLS(y.values, x.values[:, None]).fit(cov_type='cluster', cov_kwds={'groups': s.GVKEY.values})
    return r.params[0], r.bse[0], r.pvalues[0]


def convergent(d):
    rows = []
    for v in LABELS:
        s = eligible(d, [v])
        b, se, p = fe_regression(s, v)
        sd = s[v].std()
        bt = s[s.BTDUMMY == 1].set_index('BTID')[v]
        cm = s[s.BTDUMMY == 0].groupby('BTID')[v].mean()
        diff = (bt - cm.reindex(bt.index)).values
        w = stats.wilcoxon(diff)
        rows.append(dict(variable=v, label=LABELS[v], n_sets=len(bt), n_rows=len(s), n_firms=s.GVKEY.nunique(),
                         mean_bt=bt.mean(), mean_cmp=s.loc[s.BTDUMMY == 0, v].mean(), sd_sample=sd,
                         fe_b=b, fe_se_firmcl=se, fe_p=p, fe_ci_lo=b - 1.96 * se, fe_ci_hi=b + 1.96 * se,
                         fe_b_in_sd=b / sd, paired_mean_diff=diff.mean(), paired_median_diff=np.median(diff),
                         share_sets_bt_higher=np.mean(diff > 0), wilcoxon_p=w.pvalue))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- (b) discriminant: conditional logit
def pack(s, cols):
    s = s.sort_values(['BTID', 'BTDUMMY'], ascending=[True, False], kind='mergesort')
    sizes = s.groupby('BTID', sort=True).size().values
    G, M = len(sizes), sizes.max()
    gi = np.repeat(np.arange(G), sizes)
    pos = np.arange(len(s)) - np.repeat(np.cumsum(sizes) - sizes, sizes)
    X = np.zeros((G, M, len(cols)))
    mask = np.zeros((G, M), bool)
    X[gi, pos] = s[cols].values
    mask[gi, pos] = True
    bt0 = np.zeros((G, M)); bt0[gi, pos] = s['BTDUMMY'].values
    assert (bt0[:, 0] == 1).all() and (bt0[:, 1:].sum() == 0)
    firm = s.groupby('BTID', sort=True)['GVKEY'].first().values
    return X, mask, firm


def _probs(X, mask, b):
    eta = np.where(mask, X @ b, -np.inf)
    mx = eta.max(1, keepdims=True)
    w = np.exp(eta - mx)
    lse = mx[:, 0] + np.log(w.sum(1))
    return w / w.sum(1, keepdims=True), eta, lse


def clogit(X, mask, b0=None, maxit=100):
    K = X.shape[2]
    b = np.zeros(K) if b0 is None else np.asarray(b0, float).copy()
    def ll(b):
        _, eta, lse = _probs(X, mask, b)
        return np.sum(eta[:, 0] - lse)
    L = ll(b)
    for _ in range(maxit):
        p, _, _ = _probs(X, mask, b)
        xbar = np.einsum('gm,gmk->gk', p, X)
        grad = (X[:, 0, :] - xbar).sum(0)
        xc = np.where(mask[:, :, None], X - xbar[:, None, :], 0.0)
        H = np.einsum('gm,gmk,gml->kl', p, xc, xc)
        step = np.linalg.solve(H, grad)
        t = 1.0
        while True:
            Ln = ll(b + t * step)
            if Ln >= L - 1e-12 or t < 1e-6:
                break
            t /= 2
        b = b + t * step
        done = abs(Ln - L) < 1e-12 and np.max(np.abs(t * step)) < 1e-9
        L = Ln
        if done:
            break
    p, _, _ = _probs(X, mask, b)
    xbar = np.einsum('gm,gmk->gk', p, X)
    score = X[:, 0, :] - xbar
    xc = np.where(mask[:, :, None], X - xbar[:, None, :], 0.0)
    H = np.einsum('gm,gmk,gml->kl', p, xc, xc)
    return b, L, H, score


def cluster_se(H, score, firm):
    _, inv = np.unique(firm, return_inverse=True)
    S = np.zeros((inv.max() + 1, score.shape[1]))
    np.add.at(S, inv, score)
    Gf = S.shape[0]
    Hi = np.linalg.inv(H)
    V = Hi @ (S.T @ S) @ Hi * Gf / (Gf - 1)
    return np.sqrt(np.diag(V))


def concordance(pred, mask):
    """pair-weighted P(BT > control) with ties 1/2; top-1 hit rate (ties split); chance top-1"""
    p0 = pred[:, :1]
    c = mask[:, 1:]
    gt = (p0 > pred[:, 1:]) & c
    eq = (p0 == pred[:, 1:]) & c
    conc = (gt.sum() + 0.5 * eq.sum()) / c.sum()
    pm = np.where(mask, pred, -np.inf)
    mx = pm.max(1, keepdims=True)
    at_max = (pm == mx) & mask
    top1 = np.mean(at_max[:, 0] / at_max.sum(1))
    chance = np.mean(1.0 / mask.sum(1))
    return conc, top1, chance


def fit_model(s, cols, rng, raw_col=None):
    sd = s[cols].std()
    Z = s.copy()
    for c in cols:
        Z[c] = (s[c] - s[c].mean()) / sd[c]
    X, mask, firm = pack(Z, cols)
    b, L, H, score = clogit(X, mask)
    se = cluster_se(H, score, firm)
    L0 = -np.sum(np.log(mask.sum(1)))
    r2 = 1 - L / L0
    if raw_col is not None:   # single Kelly variable: rank on the raw score, higher = more BT-like
        Xr, _, _ = pack(s, [raw_col])
        pred_fn = lambda idx: Xr[idx, :, 0]
    else:
        pred_fn = None
    def stats_for(idx, bb):
        pred = pred_fn(idx) if pred_fn else X[idx] @ bb
        return concordance(pred, mask[idx])
    allidx = np.arange(X.shape[0])
    conc, top1, chance = stats_for(allidx, b)
    # firm cluster bootstrap
    firms, inv = np.unique(firm, return_inverse=True)
    sets_of = [np.where(inv == k)[0] for k in range(len(firms))]
    B = []
    for _ in range(N_BOOT):
        draw = rng.integers(0, len(firms), len(firms))
        idx = np.concatenate([sets_of[k] for k in draw])
        bb, Lb, _, _ = clogit(X[idx], mask[idx], b0=b)
        L0b = -np.sum(np.log(mask[idx].sum(1)))
        cb, tb, _ = stats_for(idx, bb)
        B.append(np.concatenate([bb, [1 - Lb / L0b, cb, tb]]))
    B = np.array(B)
    lo, hi = np.percentile(B, 2.5, 0), np.percentile(B, 97.5, 0)
    K = len(cols)
    fit = dict(n_sets=X.shape[0], n_rows=int(mask.sum()), n_firms=len(firms), k=K, LL=L, LL0=L0,
               pseudo_r2=r2, pseudo_r2_lo=lo[K], pseudo_r2_hi=hi[K],
               concordance=conc, concordance_lo=lo[K + 1], concordance_hi=hi[K + 1],
               top1_hit=top1, top1_lo=lo[K + 2], top1_hi=hi[K + 2], top1_chance=chance,
               concordance_basis='raw Kelly score' if raw_col else 'in-sample fitted index (optimistic)')
    coefs = [dict(term=c, b_per_sd=b[j], se_firmcl=se[j], p=2 * (1 - stats.norm.cdf(abs(b[j] / se[j]))),
                  odds_ratio_per_sd=math.exp(b[j]), boot_lo=lo[j], boot_hi=hi[j], sd=sd[c])
             for j, c in enumerate(cols)]
    return fit, coefs, b


def discriminant(d, rng):
    for c in ['R_LBWD', 'R_LF01', 'R_LF25', 'R_LF610', 'PCT_0_1y', 'PCT_0_5y', 'PCT_0_10y']:
        d[c + '_SQ'] = ((d[c] - d[c].mean()) / d[c].std()) ** 2
    models = {
        'M1 pctile 0-5y (primary)': (['PCT_0_5y'], 'PCT_0_5y'),
        'M1 pctile 0-1y': (['PCT_0_1y'], 'PCT_0_1y'),
        'M1 pctile 0-10y': (['PCT_0_10y'], 'PCT_0_10y'),
        'M1 resid 0-5y': (['KPST_imp_resid_0_5y'], 'KPST_imp_resid_0_5y'),
        'M2 components (bwd, fwd 0-1, fwd 2-5)': (['R_LBWD', 'R_LF01', 'R_LF25'], None),
        'UB5 upper bound, 0-5y sample': (['R_LBWD', 'R_LF01', 'R_LF25', 'R_LBWD_SQ', 'R_LF01_SQ', 'R_LF25_SQ',
                                          'PCT_0_1y', 'PCT_0_5y', 'PCT_0_1y_SQ', 'PCT_0_5y_SQ', 'TOP10_0_5y'],
                                         None),
        'UB10 upper bound, 0-10y sample': (['R_LBWD', 'R_LF01', 'R_LF25', 'R_LF610', 'R_LBWD_SQ', 'R_LF01_SQ',
                                            'R_LF25_SQ', 'R_LF610_SQ', 'PCT_0_1y', 'PCT_0_5y', 'PCT_0_10y',
                                            'PCT_0_10y_SQ', 'TOP10_0_10y'], None),
    }
    fits, coefs = [], []
    for name, (cols, raw) in models.items():
        s = eligible(d, cols)
        f, c, b = fit_model(s, cols, rng, raw)
        fits.append(dict(model=name, **f))
        coefs += [dict(model=name, **x) for x in c]
        print(f'  {name}: sets {f["n_sets"]} R2 {f["pseudo_r2"]:.4f} conc {f["concordance"]:.3f} '
              f'top1 {f["top1_hit"]:.3f} vs {f["top1_chance"]:.3f}', flush=True)
        if name.startswith('M1 pctile 0-5y'):   # cross-check against statsmodels
            from statsmodels.discrete.conditional_models import ConditionalLogit
            z = (s['PCT_0_5y'] - s['PCT_0_5y'].mean()) / s['PCT_0_5y'].std()
            r = ConditionalLogit(s['BTDUMMY'].values, z.values[:, None], groups=s['BTID'].values).fit(method='newton', maxiter=100, disp=0)
            assert abs(r.params[0] - b[0]) < 1e-6, (r.params[0], b[0])
    return pd.DataFrame(fits), pd.DataFrame(coefs)


def variance_decomp(d, rng):
    rows = []
    for v in ['PCT_0_5y', 'PCT_0_1y', 'PCT_0_10y', 'KPST_imp_resid_0_5y', 'R_LBWD', 'R_LF01', 'R_LF25']:
        for by, lab in [('BTID', 'set (BT + its 3 matched controls)'), ('GVKEY', 'firm'),
                        ('KPST_issue_year', 'grant year')]:
            s = d.dropna(subset=[v, by])
            s = s[s.groupby(by)[v].transform('size') >= 2]
            y = s[v].values
            _, g = np.unique(s[by].values, return_inverse=True)
            n_g = np.bincount(g)
            sst = np.sum((y - y.mean()) ** 2)
            def r2(yy):
                m = np.bincount(g, yy) / n_g
                return 1 - np.sum((yy - m[g]) ** 2) / sst
            obs = r2(y)
            null = np.array([r2(rng.permutation(y)) for _ in range(N_PERM)])
            G, N = len(n_g), len(y)
            msb = (sst * obs) / (G - 1); msw = (sst * (1 - obs)) / (N - G)
            k0 = (N - np.sum(n_g ** 2) / N) / (G - 1)
            row = dict(variable=v, label=LABELS[v], grouping=lab, n_rows=N, n_groups=G, r2=obs,
                       r2_null_mean=null.mean(), r2_net=obs - null.mean(),
                       icc1=(msb - msw) / (msb + (k0 - 1) * msw))
            if by == 'BTID':   # share of within-set variance explained by BT status
                ym = y - (np.bincount(g, y) / n_g)[g]
                bt = s['BTDUMMY'].values.astype(float)
                xm = bt - (np.bincount(g, bt) / n_g)[g]
                row['within_set_r2_from_btdummy'] = (xm @ ym) ** 2 / ((xm @ xm) * (ym @ ym))
            rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- (c) outcome robustness
def outcome_robustness():
    sys.dont_write_bytecode = True
    sys.path.insert(0, FD_DIR)
    import focal_diagnostics as fdg
    firm = fdg.prep_firm()
    dfm = firm.dropna(subset=fdg.FIRM_NEEDED).copy()
    dfk = dfm.dropna(subset=['KPST_imp_pctile_0_5y']).copy()
    print(f'  regression base {len(dfm)} rows / {dfm.BTID.nunique()} sets; with Kelly 0-5y {len(dfk)} / '
          f'{dfk.BTID.nunique()}', flush=True)
    K = 'KPST_imp_pctile_0_5y'
    specs = [('baseline, full N', dfm, ''), ('baseline, Kelly sample', dfk, ''),
             ('+Kelly main', dfk, f' + {K}'), ('+Kelly main + xBT', dfk, f' + {K} + {K}:BTDUMMY')]
    terms = ['BTDUMMY', 'BTDUMMY:IHSAGE2', 'BTDUMMY:IHSAGE2_C2', 'BTDUMMY:LOG5YRCT', 'BTDUMMY:IHSSALES',
             K, 'BTDUMMY:' + K]
    rows = []
    for tab in ['E (SIC2, preferred)', 'F (SIC4, preferred/headline)']:
        rhs, fe = fdg.PAPER_TABLES[tab]
        for spec, s, add in specs:
            res, n, npos = fdg.fit_three(s, rhs + add, fe)
            for est in fdg.EST[:3]:
                r = res[est]
                for t in terms:
                    if t in r.index:
                        rows.append(dict(table=tab, estimator=est, spec=spec, term=t, b=r.loc[t, 'b'],
                                         se=r.loc[t, 'se'], p=r.loc[t, 'p'],
                                         n=npos if est.startswith('ZT') else n, n_sets=s.BTID.nunique()))
            print(f'  {tab} | {spec} done', flush=True)
    long = pd.DataFrame(rows)
    base = long[long.spec == 'baseline, Kelly sample'].set_index(['table', 'estimator', 'term'])[['b', 'se']]
    long = long.join(base.rename(columns={'b': 'b_base_same_sample', 'se': 'se_base_same_sample'}),
                     on=['table', 'estimator', 'term'])
    long['delta_b_vs_same_sample'] = long.b - long.b_base_same_sample
    long['delta_in_base_se'] = long.delta_b_vs_same_sample / long.se_base_same_sample
    # reproduction check of the full-N baseline against the earlier verified run
    ref = pd.read_csv(os.path.join(FD_DIR, 'step3_paper_tables_all_samples_long.csv'))
    ref = ref[ref['sample'] == 'full'].set_index(['table', 'estimator', 'term']).b
    chk = long[long.spec == 'baseline, full N'].join(ref.rename('b_ref'), on=['table', 'estimator', 'term'])
    print(f'  max |b - verified paper reproduction| on full-N baseline: {np.nanmax(np.abs(chk.b - chk.b_ref)):.2e}')
    return long


def main():
    only = set(sys.argv[1:])
    run = lambda k: (not only) or (k in only)
    rng = np.random.default_rng(SEED)
    d = load()
    if run('a'):
        coverage(d).to_csv(out('test1_coverage.csv'), index=False)
        conv = convergent(d)
        conv.to_csv(out('test1_convergent.csv'), index=False)
        print(conv[['variable', 'n_sets', 'mean_bt', 'mean_cmp', 'fe_b', 'fe_se_firmcl', 'fe_p', 'fe_b_in_sd',
                    'share_sets_bt_higher', 'wilcoxon_p']].round(4).to_string(index=False))
    if run('b'):
        fits, coefs = discriminant(d, rng)
        fits.to_csv(out('test1_clogit_fit.csv'), index=False)
        coefs.to_csv(out('test1_clogit_coefs.csv'), index=False)
        vd = variance_decomp(d, rng)
        vd.to_csv(out('test1_variance_decomp.csv'), index=False)
        print(vd[['variable', 'grouping', 'r2', 'r2_null_mean', 'r2_net', 'icc1',
                  'within_set_r2_from_btdummy']].round(4).to_string(index=False))
    if run('c'):
        outcome_robustness().to_csv(out('test1_outcome_robustness.csv'), index=False)


if __name__ == '__main__':
    main()
