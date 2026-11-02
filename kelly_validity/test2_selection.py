"""
Test 2 -- sample selection vs the patent population on Kelly, Papanikolaou, Seru &
Taddy (2021) text-based importance. Exploit paper, firm sample only.

For breakthroughs (BTDUMMY=1) and controls (BTDUMMY=0) separately, each sample
patent is matched to random population patents from the same stratum:
  (i)   grant year
  (ii)  grant year x primary USPC class      (primary benchmark)
  (iii) grant year x primary USPC subclass
1,000 replicate draws (one random pool patent per sample patent per draw) give the
null distribution of mean pctile, top-10% share and median pctile under the
hypothesis that sample patents are random draws from their strata. The pool
excludes all 6,576 sample patents. A one-sample KS test compares the observed
pctile distribution with the exact stratum-mixture population CDF.

Windows: 0_5y (primary), 0_1y and 0_10y (robustness).
Top 10% = imp_pctile >= 0.90 (pooled 1836-2010 percentile, as in KPST).

Run:  python test2_selection.py           (~5-10 min)
"""
import os
import time

import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.parquet as pq
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)
SAMPLE_CSV = os.path.join(BASE, 'bt_firm_dataset_segments_20262104c_with_topic_size_kpst_focal.csv')
KELLY = r'C:\Users\ensig\Kelly et al Replication'
KPST_PQ = os.path.join(KELLY, 'Measuring-Technological-Innovation-Over-the-Long-Run-Replication-Kit-master',
                       'novelty_scores', 'patent_novelty_scores.parquet')
USPC_PQ = os.path.join(KELLY, 'variance_decomp', 'data', 'patents.parquet')

WINDOWS = ['0_5y', '0_1y', '0_10y']
TIERS = ['year', 'year_x_class', 'year_x_subclass']
TOP = 0.90
R = 1000
CHUNK = 200
SEED = 20261029
Y0, Y1 = 1989, 2010


def out(name):
    return os.path.join(HERE, name)


def load():
    t0 = time.time()
    s = pd.read_csv(SAMPLE_CSV, usecols=['BTID', 'PATENT', 'BTDUMMY', 'KPST_imp_pctile_0_5y'])
    s['PATENT'] = s['PATENT'].astype(np.int64)

    cols = ['patent_number', 'issue_year'] + [f'imp_pctile_{w}' for w in WINDOWS] + \
           ['breakthrough_top10_0_5y', 'breakthrough_top10_0_10y']
    k = pq.read_table(KPST_PQ, columns=cols).to_pandas()
    k['patent_number'] = k['patent_number'].astype(np.int64)

    t = pq.read_table(USPC_PQ, columns=['pid_num', 'id_kind', 'class_primary', 'subclass_primary'])
    t = t.filter(pc.equal(t['id_kind'], 'utility'))
    cls = pc.dictionary_encode(t['class_primary']).combine_chunks()
    sub = pc.dictionary_encode(t['subclass_primary']).combine_chunks()
    u = pd.DataFrame({
        'patent_number': t['pid_num'].to_numpy().astype(np.int64),
        'cls': cls.indices.to_numpy(zero_copy_only=False),
        'sub': sub.indices.to_numpy(zero_copy_only=False),
    })
    u['cls'] = u['cls'].fillna(-1).astype(np.int64)
    u['sub'] = u['sub'].fillna(-1).astype(np.int64)
    del t
    k = k.merge(u, on='patent_number', how='left')
    k['cls'] = k['cls'].fillna(-1).astype(np.int64)
    k['sub'] = k['sub'].fillna(-1).astype(np.int64)
    print(f'loaded KPST {len(k):,} rows, USPC utility {len(u):,} rows in {time.time()-t0:.0f}s')
    return s, k


def checks(s, k):
    rows = []
    ks = k.set_index('patent_number')
    in_k = s.PATENT.isin(ks.index)
    rows.append(('sample patents', len(s)))
    rows.append(('sample patents in KPST universe', int(in_k.sum())))
    m = s[in_k].join(ks, on='PATENT')
    rows.append(('sample patents with USPC primary class', int((m['cls'] >= 0).sum())))
    rows.append(('sample patents with USPC primary subclass', int((m['sub'] >= 0).sum())))
    both = m['KPST_imp_pctile_0_5y'].notna() & m['imp_pctile_0_5y'].notna()
    diff = (m.loc[both, 'KPST_imp_pctile_0_5y'] - m.loc[both, 'imp_pctile_0_5y']).abs().max()
    rows.append(('max |pctile_0_5y file - parquet|', float(diff)))
    pop = k[(k.issue_year >= Y0) & (k.issue_year <= Y1)]
    rows.append((f'population patents {Y0}-{Y1} (KPST)', len(pop)))
    rows.append((f'population {Y0}-{Y1} with USPC primary class', int((pop['cls'] >= 0).sum())))
    for w in ['0_5y', '0_10y']:
        sc = pop[f'imp_pctile_{w}'].notna()
        agree = ((pop.loc[sc, f'imp_pctile_{w}'] >= TOP).astype(int) ==
                 pop.loc[sc, f'breakthrough_top10_{w}']).mean()
        rows.append((f'agreement pctile>=0.90 vs KPST top10 flag, {w}', float(agree)))
    return pd.DataFrame(rows, columns=['check', 'value'])


def baseline(k):
    pop = k[(k.issue_year >= Y0) & (k.issue_year <= Y1)]
    rows, by_year = [], []
    for w in WINDOWS:
        c = f'imp_pctile_{w}'
        v = pop[c].dropna()
        rows.append(dict(window=w, n=len(v), mean_pctile=v.mean(), median_pctile=v.median(),
                         sd_pctile=v.std(), top10_share=(v >= TOP).mean(),
                         first_year=int(pop.loc[pop[c].notna(), 'issue_year'].min()),
                         last_year=int(pop.loc[pop[c].notna(), 'issue_year'].max())))
        g = pop[pop[c].notna()].groupby('issue_year')[c]
        yy = pd.DataFrame({'n': g.size(), 'mean_pctile': g.mean(), 'median_pctile': g.median(),
                           'top10_share': g.apply(lambda x: (x >= TOP).mean())}).reset_index()
        yy.insert(0, 'window', w)
        by_year.append(yy)
    return pd.DataFrame(rows), pd.concat(by_year)


def strata_key(df, tier):
    y = df['issue_year'].to_numpy().astype(np.int64)
    if tier == 'year':
        return y, np.ones(len(df), bool)
    c = df['cls' if tier == 'year_x_class' else 'sub'].to_numpy()
    return y * 10_000_000 + c, c >= 0


def stat_block(x):
    """x: (reps, n) -> mean, top10 share, median per rep."""
    return x.mean(axis=1), (x >= TOP).mean(axis=1), np.median(x, axis=1)


def mixture_cdf_ks(obs, obs_keys, pool_vals_sorted, starts, counts, key_index):
    """One-sample KS of obs against F(x) = mean_i F_{stratum(i)}(x)."""
    n = len(obs)
    xs = np.sort(obs)
    uk, nk = np.unique(obs_keys, return_counts=True)
    F = np.zeros(n)
    F_left = np.zeros(n)
    for key, wcount in zip(uk, nk):
        j = key_index[key]
        seg = pool_vals_sorted[starts[j]:starts[j] + counts[j]]
        F += wcount * np.searchsorted(seg, xs, side='right') / counts[j]
        F_left += wcount * np.searchsorted(seg, xs, side='left') / counts[j]
    F /= n
    F_left /= n
    ecdf_hi = np.arange(1, n + 1) / n
    ecdf_lo = np.arange(0, n) / n
    d = max(np.max(ecdf_hi - F), np.max(F_left - ecdf_lo))
    p = stats.kstwo.sf(d, n)
    return d, p


def subclass_matchable(s, k, w):
    """Sample patents whose grant year x subclass stratum has >=1 non-sample pool patent."""
    c = f'imp_pctile_{w}'
    pop = k[k[c].notna() & ~k.patent_number.isin(set(s.PATENT)) & k.issue_year.between(Y0, Y1)]
    pk, ok = strata_key(pop, 'year_x_subclass')
    m = s.join(k.set_index('patent_number'), on='PATENT')
    m = m[m[c].notna()]
    sk, sok = strata_key(m, 'year_x_subclass')
    keep = sok & np.isin(sk, np.unique(pk[ok]))
    return set(m.loc[keep, 'PATENT'])


def run_tier(s, k, w, tier, rng, subset='all', exclude=None):
    c = f'imp_pctile_{w}'
    sample_ids = set(s.PATENT.tolist()) if exclude is None else exclude
    pop = k[k[c].notna() & ~k.patent_number.isin(sample_ids) & k.issue_year.between(Y0, Y1)]
    pkey, pok = strata_key(pop, tier)
    pkey, pval = pkey[pok], pop[c].to_numpy()[pok].astype(np.float64)
    order = np.lexsort((pval, pkey))
    pkey, pval = pkey[order], pval[order]
    ukeys, starts, counts = np.unique(pkey, return_index=True, return_counts=True)
    key_index = {kk: i for i, kk in enumerate(ukeys)}

    ks_ = k.set_index('patent_number')
    m = s.join(ks_, on='PATENT')
    rows, cov = [], []
    for grp, flag in [('BT', 1), ('CMP', 0)]:
        g = m[(m.BTDUMMY == flag)]
        n_all = len(g)
        g = g[g[c].notna()]
        n_scored = len(g)
        skey, sok = strata_key(g, tier)
        g, skey = g[sok], skey[sok]
        n_classed = len(g)
        has = np.isin(skey, ukeys)
        g, skey = g[has], skey[has]
        obs = g[c].to_numpy().astype(np.float64)
        n = len(obs)
        cov.append(dict(window=w, tier=tier, subset=subset, group=grp, n_rows=n_all, n_scored=n_scored,
                        n_with_stratum_info=n_classed, n_no_pool_match=n_classed - n,
                        n_used=n, median_pool_size=float(np.median(counts[[key_index[x] for x in skey]]))))
        idx_s = np.array([key_index[x] for x in skey])
        st, ct = starts[idx_s], counts[idx_s]
        nulls = [[], [], []]
        draws_sum, draws_sq, draws_n = 0.0, 0.0, 0
        for r0 in range(0, R, CHUNK):
            reps = min(CHUNK, R - r0)
            u = rng.random((reps, n))
            x = pval[st + np.minimum((u * ct).astype(np.int64), ct - 1)]
            for i, a in enumerate(stat_block(x)):
                nulls[i].append(a)
            draws_sum += x.sum(); draws_sq += (x * x).sum(); draws_n += x.size
        nulls = [np.concatenate(a) for a in nulls]
        pop_mean = draws_sum / draws_n
        pop_sd = np.sqrt(draws_sq / draws_n - pop_mean ** 2)
        ks_d, ks_p = mixture_cdf_ks(obs, skey, pval, starts, counts, key_index)
        observed = [obs.mean(), (obs >= TOP).mean(), np.median(obs)]
        for name, o, nd in zip(['mean_pctile', 'top10_share', 'median_pctile'], observed, nulls):
            p_hi = (1 + np.sum(nd >= o)) / (R + 1)
            p_lo = (1 + np.sum(nd <= o)) / (R + 1)
            rows.append(dict(window=w, tier=tier, subset=subset, group=grp, stat=name, n_used=n,
                             observed=o, null_mean=nd.mean(), null_lo95=np.quantile(nd, 0.025),
                             null_hi95=np.quantile(nd, 0.975), p_two_sided=min(1.0, 2 * min(p_hi, p_lo)),
                             diff=o - nd.mean(),
                             diff_pctile_pts=(o - nd.mean()) * 100 if name != 'top10_share' else np.nan,
                             diff_share_pts=(o - nd.mean()) * 100 if name == 'top10_share' else np.nan,
                             diff_in_pop_sd=(o - nd.mean()) / pop_sd if name == 'mean_pctile' else np.nan,
                             pop_sd=pop_sd, ks_D=ks_d, ks_p=ks_p))
    return rows, cov


def main():
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    s, k = load()
    chk = checks(s, k)
    chk.to_csv(out('test2_checks.csv'), index=False)
    print(chk.to_string(index=False))
    b, by = baseline(k)
    b.to_csv(out('test2_population_baseline.csv'), index=False)
    by.to_csv(out('test2_population_baseline_by_year.csv'), index=False)
    print(b.to_string(index=False))
    allrows, allcov = [], []
    for w in WINDOWS:
        for tier in TIERS:
            r, c = run_tier(s, k, w, tier, rng)
            allrows += r; allcov += c
            print(f'{w} {tier} done {time.time()-t0:.0f}s')
        # same patents across all tiers: those with a year x subclass pool match
        keep = subclass_matchable(s, k, w)
        s_common = s[s.PATENT.isin(keep)]
        for tier in TIERS[:2]:
            r, c = run_tier(s_common, k, w, tier, rng, subset='subclass_matchable',
                            exclude=set(s.PATENT))
            allrows += r; allcov += c
        print(f'{w} common-subset tiers done {time.time()-t0:.0f}s')
    res = pd.DataFrame(allrows)
    cov = pd.DataFrame(allcov)
    res.to_csv(out('test2_benchmark.csv'), index=False)
    cov.to_csv(out('test2_coverage.csv'), index=False)
    pd.set_option('display.width', 250)
    print(cov.to_string(index=False))
    print(res[['window', 'tier', 'group', 'stat', 'n_used', 'observed', 'null_mean', 'null_lo95',
               'null_hi95', 'p_two_sided', 'diff_in_pop_sd', 'ks_D', 'ks_p']].to_string(index=False))
    print(f'done {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
