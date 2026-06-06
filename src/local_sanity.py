"""Local sanity check of the PURE numpy/python parts of modal_app.py (no torch/modal exec).
Verifies the critical-bug fixes: regime_switch finite, dial monotone & deconfounded, edge cases.
Run: py -3.13 local_sanity.py"""
import math
import numpy as np
import modal_app as M

ok = True


def check(name, cond):
    global ok
    print(("PASS" if cond else "FAIL"), name)
    ok = ok and cond


# 1. Generators produce finite, sane returns
for gen in ["gbm", "gjr_garch", "regime_switch"]:
    r = M.oracle_returns(gen, 50000, seed=0)
    check(f"{gen} finite", np.isfinite(r).all())
    check(f"{gen} sane std (<0.2)", 1e-5 < r.std() < 0.2)
    f = M.stylised_facts(r)
    print(f"   {gen}: std={r.std():.4f} kurt={f['excess_kurtosis']:.2f} "
          f"alpha={f['tail_index']:.2f} clust={f['vol_clustering']:.3f} lev={f['leverage']:.3f}")

# 2. GJR-GARCH shows fat tails + clustering + leverage vs GBM
rg = M.oracle_returns("gjr_garch", 100000, seed=1)
fg = M.stylised_facts(rg)
check("garch fat tails (kurt>1)", fg["excess_kurtosis"] > 1.0)
check("garch clustering (>0.02)", fg["vol_clustering"] > 0.02)
check("garch leverage (<0)", fg["leverage"] < 0.0)

# 3. regime_switch is finite at Oracle AND non-stationary (clustering present)
rr = M.oracle_returns("regime_switch", 100000, seed=2)
fr = M.stylised_facts(rr)
check("regime finite + clustering", np.isfinite(rr).all() and fr["vol_clustering"] > 0.02)

# 4. Fidelity dial: monotone-ish fidelity in lambda, deconfounded variance
for gen in ["gjr_garch", "regime_switch"]:
    target = M.stylised_facts(M.oracle_returns(gen, 300000, seed=777))
    fids, stds = [], []
    for lam in [0.0, 0.25, 0.5, 0.75, 1.0]:
        d = M.dial_returns(gen, lam, 100000, seed=3)
        fids.append(M.fidelity_score(M.stylised_facts(d), target))
        stds.append(float(d.std()))
    print(f"   {gen} dial fids={[round(x,3) for x in fids]} stds={[round(x,4) for x in stds]}")
    check(f"{gen} dial lam=1 high fidelity (>0.6)", fids[-1] > 0.6)
    check(f"{gen} dial lam=0 lower than lam=1", fids[0] < fids[-1])
    check(f"{gen} dial variance deconfounded (std ratio<2)", max(stds) / min(stds) < 2.0)

# 5. financial_metrics edge cases (no crash)
check("fin empty", M.financial_metrics([])["sharpe"] == 0.0)
check("fin len1", M.financial_metrics([0.01])["sharpe"] == 0.0)
fm = M.financial_metrics(np.random.default_rng(0).normal(0.0005, 0.01, 300))
check("fin sane sharpe", abs(fm["sharpe"]) < 10 and np.isfinite(fm["sharpe"]))

# 6. hill_tail_index guard on degenerate input
check("hill degenerate no crash", math.isnan(M.hill_tail_index(np.zeros(1000))) or True)

# 7. _make_windows + segments
X, y = M._make_windows(np.arange(10.0), 3)
check("windows causal shape", X.shape == (7, 3) and y[0] == 3.0 and X[0, 0] == 0.0)
Xs, ys = M._make_windows_segments([np.arange(6.0), np.arange(6.0)], 3)
check("segment windows no straddle", len(Xs) == 6)  # 3 per segment, none across boundary

# 8. analyze.py helpers
import analyze as A
check("norm_ppf/cdf inverse", abs(A.norm_cdf(A.norm_ppf(0.975)) - 0.975) < 1e-3)
sl, _ = A.ols_slope([0, 1, 2, 3], [0, 2, 4, 6])
check("ols slope", abs(sl - 2.0) < 1e-6)

# 9. Deflated Sharpe per-step units (not saturated for a modest best)
psr, sr0 = A.deflated_sharpe([1.0, 0.8, 1.2, 0.5, 0.9, 1.1, 0.7, 1.3], T=2000)
check("DSR in (0,1) not saturated", psr is not None and 0.0 < psr < 1.0)
print(f"   DSR psr={psr:.3f} sr0_ann={sr0:.3f}")

# 10. analyze end-to-end on a tiny fake ledger (headline filter + a1 filter + date-aligned panel)
import json, tempfile, os
fake = []
for fid in [0.30, 0.50, 0.70, 0.32, 0.52, 0.72]:
    fake.append({"task": "dial_transfer", "gen": "gjr_garch", "headline": True, "fidelity": fid,
                 "transfer_sharpe": fid * 1.5 - 0.2, "gap": 0.3, "random_sharpe": -0.1,
                 "n_eval": 2000, "cost_bps": 5.0})
fake.append({"task": "dial_transfer", "gen": "gjr_garch", "fidelity": 0.9, "transfer_sharpe": 9.9,
             "cost_bps": 0.0})  # exp09 cost-ablation, NO headline -> must be excluded
for s in range(3):
    fake.append({"task": "learned_transfer", "gen": "gjr_garch", "a1": True, "modelfree_sharpe": 0.4,
                 "world_model_bias": 0.2, "dream_sharpe": 0.2, "kind": "transformer_mdn", "n_bars": 60000})
# a contaminating exp07 modelfree row (different sharpe) that must NOT enter the TOST reference
fake.append({"task": "learned_transfer", "gen": "gjr_garch", "modelfree_sharpe": 5.0,
             "world_model_bias": 9.0, "kind": "transformer_mdn", "n_bars": 2000})
# real_transfer with date-keyed strat series for the panel bootstrap
per = [{"ticker": t, "dream_sharpe": 0.1, "modelfree_sharpe": 0.0, "buyhold_sharpe": 0.3,
        "dates": [f"2024-01-{d:02d}" for d in range(1, 28)],
        "dream_strat": list(np.random.default_rng(i).normal(0, 0.01, 27))} for i, t in enumerate(["A", "B", "C"])]
fake.append({"task": "real_transfer", "per_ticker": per})
tmp = os.path.join(tempfile.gettempdir(), "fake_ledger.jsonl")
with open(tmp, "w") as f:
    for r in fake:
        f.write(json.dumps(r) + "\n")
loaded = A.load(tmp)
v = A.verdict_for(loaded, "gjr_garch", [0.4, 0.4, 0.4])
check("analyze excludes exp09 (6 headline pts)", v.get("n_points") == 6)
check("analyze verdict decided", v.get("verdict") in
      ("necessary_not_sufficient", "sufficient", "decoupled", "partial_necessary", "inconclusive"))
pb = A.panel_bootstrap_real(loaded)
check("panel bootstrap date-aligned runs", pb is not None and pb["n_common_dates"] == 27 and pb["effective_n"] >= 1.0)
print(f"   verdict={v.get('verdict')} slope={v.get('slope'):.2f} n={v.get('n_points')} panel_effN={pb['effective_n']:.2f}")

# 11. H3 decomposition excludes a1 (exp08) rows from the learning curve (no cross-experiment contamination)
h3 = A.h3_decomposition(loaded)
check("h3 learning curve excludes a1 n_bars=60000", 60000 not in (h3.get("learning_curve") or {}))
check("h3 world_model_bias per-gen present", "world_model_bias_by_gen" in h3)
print(f"   h3 learning_curve={h3.get('learning_curve')} wmb_by_gen={h3.get('world_model_bias_by_gen')}")

# 12. momentum generator: a learnable directional edge rising with snr; still carries stylized facts
r0 = M.momentum_returns(0.0, 60000, seed=1); r4 = M.momentum_returns(4.0, 60000, seed=1)
e0 = M.directional_r2(r0, 60); e4 = M.directional_r2(r4, 60)
f4 = M.stylised_facts(r4)
check("momentum snr=0 negligible edge", e0 < 0.03)
check("momentum snr=4 learnable edge > snr=0", e4 > e0 and e4 > 0.05)
check("momentum carries stylized facts", f4["excess_kurtosis"] > 1 and f4["vol_clustering"] > 0.02)
print(f"   momentum: r2(snr0)={e0:.4f} r2(snr4)={e4:.4f} kurt={f4['excess_kurtosis']:.1f} clust={f4['vol_clustering']:.3f}")

print("\nALL PASS" if ok else "\nSOME FAILED")
import sys
sys.exit(0 if ok else 1)
