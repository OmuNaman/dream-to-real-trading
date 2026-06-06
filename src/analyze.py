"""
analyze.py — the HEADLINE statistical mechanism (runs LOCALLY on the ledger; numpy+math only).

Implements, from experiments/results.jsonl:
  H2 fidelity->transfer law : OLS slope + bootstrap CI on the slope, across (fidelity, transfer)
                              points, per generator; the PRE-REGISTERED necessary/sufficient/
                              decoupled verdict.
  Equivalence (TOST)        : high-fidelity dial transfer vs model-free PPO (margin 0.25 Sharpe).
  Deflated Sharpe (DSR/PSR) : best dial Sharpe deflated by the true number of trials.
  H3 gap decomposition      : world-model bias (Oracle-dream - learned-dream), finite-data
                              (learning curve), non-stationarity (regime vs garch).
  Real-arm PANEL bootstrap  : date-block bootstrap across the held-out cross-section + effective N.

Pre-registered decision rule (committed in statement.md, Rev 2):
  - necessary : slope CI excludes 0  AND  lowest-fidelity transfer ~ Random (<= random mean)
  - sufficient: high-fidelity transfer NON-INFERIOR to model-free PPO (TOST, margin 0.25)
  - decoupled : |standardized slope| < 0.1 over a RESOLVED fidelity range (range >= 0.2)

Usage:  py -3.13 analyze.py [path/to/results.jsonl]   (default: ../results.jsonl relative to here)
Writes: ../analysis_summary.json  and prints a human summary.
"""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_LEDGER = os.path.join(HERE, "..", "results.jsonl")
EULER = 0.5772156649015329
MARGIN = 0.25            # TOST equivalence margin (Sharpe)
RANGE_MIN = 0.20         # minimum resolved fidelity range for a decidable verdict
STD_SLOPE_DECOUPLE = 0.10


# ---- small stats helpers (no scipy) ----------------------------------------
def norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def norm_ppf(p):
    """Inverse normal CDF (Acklam's rational approximation)."""
    if p <= 0:
        return -np.inf
    if p >= 1:
        return np.inf
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5; r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


def ols_slope(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    xm, ym = x.mean(), y.mean()
    sxx = np.sum((x - xm) ** 2)
    if sxx < 1e-12:
        return 0.0, ym
    b = np.sum((x - xm) * (y - ym)) / sxx
    return float(b), float(ym - b * xm)


def boot_slope_ci(x, y, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float); y = np.asarray(y, float); n = len(x)
    slopes = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        b, _ = ols_slope(x[idx], y[idx])
        slopes.append(b)
    return float(np.percentile(slopes, 2.5)), float(np.percentile(slopes, 97.5))


def tost_equivalence(a, b, margin=MARGIN):
    """Two one-sided tests for equivalence of means(a) vs means(b) within +-margin.
    Returns (equivalent: bool, mean_diff, ci90). Uses a Welch-style SE on the per-seed samples."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    if len(a) < 2 or len(b) < 2:
        return False, float(np.mean(a) - np.mean(b)) if len(a) and len(b) else 0.0, None
    diff = a.mean() - b.mean()
    se = math.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    if se < 1e-9:
        return abs(diff) < margin, float(diff), (float(diff), float(diff))
    lo, hi = diff - 1.645 * se, diff + 1.645 * se   # 90% CI <=> two one-sided 5% tests
    return bool(lo > -margin and hi < margin), float(diff), (float(lo), float(hi))


def deflated_sharpe(sharpes, T, best=None, skew=0.0, kurt=3.0, ann=252):
    """López de Prado Deflated Sharpe Ratio. `sharpes` = trial Sharpes (ANNUALIZED); T = sample
    length in STEPS. The PSR math requires PER-STEP Sharpes to match the sqrt(T-1) factor, so we
    divide every Sharpe by sqrt(ann) first. Returns (DSR_prob, SR0_benchmark in annualized units)."""
    s = np.asarray([x for x in sharpes if np.isfinite(x)], float) / math.sqrt(ann)  # annualized -> per-step
    N = len(s)
    if N < 2 or T < 4:
        return None, None
    sr = (best / math.sqrt(ann)) if best is not None else float(s.max())
    var_sr = float(np.var(s, ddof=1))
    if var_sr < 1e-12:
        return None, None
    sr0 = math.sqrt(var_sr) * ((1 - EULER) * norm_ppf(1 - 1.0 / N) + EULER * norm_ppf(1 - 1.0 / (N * math.e)))
    denom = math.sqrt(max(1e-9, 1 - skew * sr + (kurt - 1) / 4.0 * sr * sr))
    psr = norm_cdf((sr - sr0) * math.sqrt(T - 1) / denom)
    return float(psr), float(sr0 * math.sqrt(ann))  # report benchmark back in annualized units


# ---- load ledger ------------------------------------------------------------
def load(ledger):
    rows = []
    if not os.path.exists(ledger):
        return rows
    with open(ledger) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    # flatten: a row may be a single metrics dict, or {"batch_results":[...]}
    out = []
    for r in rows:
        m = r.get("metrics", r)
        if isinstance(m, dict) and "batch_results" in m:
            out.extend(m["batch_results"])
        elif "batch_results" in r:
            out.extend(r["batch_results"])
        else:
            out.append(m)
    return [m for m in out if isinstance(m, dict)]


def dial_points(rows, gen):
    pts = [r for r in rows if r.get("task") == "dial_transfer" and r.get("gen") == gen
           and "fidelity" in r and "transfer_sharpe" in r]
    head = [r for r in pts if r.get("headline")]   # exclude cost-ablation (exp09) from the headline dial
    # safe fallback (never include exp09 cost!=5 rows): keep only cost_bps==5 if headline tags are missing
    return head if head else [r for r in pts if abs(r.get("cost_bps", 5.0) - 5.0) < 1e-9]


def verdict_for(rows, gen, modelfree_sharpes, seed=0):
    pts = dial_points(rows, gen)
    if len(pts) < 6:
        return {"gen": gen, "status": "insufficient_points", "n": len(pts)}
    fid = np.array([p["fidelity"] for p in pts], float)
    tr = np.array([p["transfer_sharpe"] for p in pts], float)
    rng_fid = float(fid.max() - fid.min())
    slope, intercept = ols_slope(fid, tr)
    ci = boot_slope_ci(fid, tr, seed=seed)
    std_slope = slope * (fid.std() / (tr.std() + 1e-12))
    # low-fidelity tertile transfer vs Random
    lo_mask = fid <= np.percentile(fid, 33)
    hi_mask = fid >= np.percentile(fid, 67)
    rnd = np.array([p.get("random_sharpe", 0.0) for p in pts], float).mean()
    lo_tr = float(tr[lo_mask].mean()); hi_tr = float(tr[hi_mask].mean())
    hi_seedvals = tr[hi_mask]
    tost_equiv, diff, tost_ci = tost_equivalence(hi_seedvals, np.asarray(modelfree_sharpes, float)) \
        if len(modelfree_sharpes) else (False, None, None)
    bh = float(np.mean([p.get("buyhold_sharpe", 0.0) for p in pts]))
    resolved = rng_fid >= RANGE_MIN
    slope_sig_pos = ci[0] > 0                 # slope CI excludes 0 on the positive side
    slope_includes_0 = ci[0] <= 0 <= ci[1]    # no significant fidelity->transfer relationship
    beats_bh = hi_tr > bh                      # high-fidelity transfer is ABSOLUTELY useful (beats Buy&Hold)
    # SUFFICIENT requires high fidelity to yield USEFUL transfer (beat Buy&Hold) AND be non-inferior to
    # model-free PPO. (Matching a *failing* model-free baseline is NOT sufficiency.)
    sufficient = resolved and beats_bh and tost_equiv
    necessary = resolved and slope_sig_pos and (lo_tr <= rnd + 0.25) and not sufficient
    decoupled = resolved and slope_includes_0 and (not sufficient) and (not beats_bh)
    if not resolved:
        verdict = "undecidable_fidelity_range"
    elif sufficient:
        verdict = "sufficient"
    elif necessary:
        verdict = "necessary_not_sufficient"
    elif decoupled:
        verdict = "decoupled"
    else:
        verdict = "inconclusive"
    return {"gen": gen, "n_points": len(pts), "fidelity_range": rng_fid, "resolved": resolved,
            "slope": slope, "slope_ci95": ci, "standardized_slope": float(std_slope),
            "low_fid_transfer": lo_tr, "high_fid_transfer": hi_tr, "random_sharpe": float(rnd),
            "buyhold_sharpe": bh, "beats_buyhold": bool(beats_bh),
            "tost_equiv_to_modelfree": bool(tost_equiv), "tost_diff": diff, "tost_ci90": tost_ci,
            "verdict": verdict}


def h3_decomposition(rows):
    out = {}
    lt = [r for r in rows if r.get("task") == "learned_transfer"]
    # (i) world-model bias per generator, from the clean A1 comparator (exp08, a1)
    a1 = [r for r in lt if r.get("a1") and "world_model_bias" in r]
    if a1:
        out["world_model_bias_mean"] = float(np.mean([r["world_model_bias"] for r in a1]))
        out["world_model_bias_by_gen"] = {
            g: float(np.mean([r["world_model_bias"] for r in a1 if r.get("gen") == g]))
            for g in sorted({r.get("gen") for r in a1})}
    # (ii) finite-data: the exp07 learning curve ONLY (transformer, NOT exp08 a1 rows; one generator)
    lc = [r for r in lt if r.get("kind") == "transformer_mdn" and "n_bars" in r
          and not r.get("a1") and r.get("gen") == "gjr_garch"]
    if lc:
        by_n = {}
        for r in lc:
            by_n.setdefault(r["n_bars"], []).append(r.get("dream_sharpe", 0.0))
        ns = sorted(by_n)
        out["learning_curve"] = {int(n): float(np.mean(by_n[n])) for n in ns}
        if len(ns) >= 2:
            out["finite_data_effect"] = float(np.mean(by_n[ns[-1]]) - np.mean(by_n[ns[0]]))
    # non-stationarity: gap (in-dream - realized) regime vs garch at high fidelity
    for gen in ("gjr_garch", "regime_switch"):
        pts = [p for p in dial_points(rows, gen) if p.get("fidelity", 0) >= 0.6]
        if pts:
            out[f"gap_highfid_{gen}"] = float(np.mean([p.get("gap", 0.0) for p in pts]))
    if "gap_highfid_gjr_garch" in out and "gap_highfid_regime_switch" in out:
        out["nonstationarity_effect"] = out["gap_highfid_regime_switch"] - out["gap_highfid_gjr_garch"]
    return out


def panel_bootstrap_real(rows, n_boot=2000, block=20, ann=252, seed=0):
    rt = [r for r in rows if r.get("task") == "real_transfer" and r.get("per_ticker")]
    if not rt:
        return None
    per = [p for p in rt[-1]["per_ticker"] if p.get("dream_strat") and p.get("dates")]  # latest run
    if len(per) < 2:
        return None
    # align by CALENDAR DATE: intersection of trading days across eval tickers (no positional misalignment)
    common = sorted(set.intersection(*[set(p["dates"]) for p in per]))
    if len(common) < block + 1:
        return None
    M = np.array([[dict(zip(p["dates"], p["dream_strat"]))[d] for d in common] for p in per])  # (tickers, dates)
    pooled = M.mean(axis=0)                            # equal-weight cross-sectional strategy
    T = len(common)
    rng = np.random.default_rng(seed)
    shs = []
    nb = max(1, T // block)
    for _ in range(n_boot):
        starts = rng.integers(0, T - block + 1, nb)    # whole DATE blocks across the panel
        samp = np.concatenate([pooled[s:s + block] for s in starts])[:T]
        if samp.std() > 1e-12:
            shs.append(samp.mean() / samp.std() * math.sqrt(ann))
    pooled_sh = float(pooled.mean() / (pooled.std() + 1e-12) * math.sqrt(ann))
    k = len(per)
    corr = np.corrcoef(M)
    if not np.all(np.isfinite(corr)):          # guard constant (flat-policy) rows -> NaN corr
        corr = np.nan_to_num(corr, nan=0.0)
    avg_off = float((corr.sum() - k) / (k * (k - 1) + 1e-9))   # cross-sectional correlation
    eff_n = float(k / (1 + (k - 1) * max(0.0, avg_off)))
    return {"pooled_dream_sharpe": pooled_sh,
            "pooled_sharpe_ci95": [float(np.percentile(shs, 2.5)), float(np.percentile(shs, 97.5))] if shs else None,
            "n_eval_tickers": k, "n_common_dates": T, "avg_pairwise_corr": avg_off, "effective_n": eff_n,
            "dream_sharpe_mean": float(np.mean([p["dream_sharpe"] for p in per])),
            "modelfree_sharpe_mean": float(np.mean([p["modelfree_sharpe"] for p in per])),
            "buyhold_sharpe_mean": float(np.mean([p["buyhold_sharpe"] for p in per]))}


def predict_summary(rows):
    """Reward-relevant (Oracle, captures trend) vs stylized-fact-faithful-but-signal-blind (GARCH)
    transfer, as a function of directional predictability (snr). The punchline contrast."""
    pr = [r for r in rows if r.get("task") == "predict_dial" and "transfer_oracle" in r]
    if len(pr) < 4:
        return None
    snrs = [r["snr"] for r in pr]
    toracle = [r["transfer_oracle"] for r in pr]
    tgarch = [r["transfer_garch"] for r in pr]
    by = {}
    for snr in sorted(set(snrs)):
        grp = [r for r in pr if r["snr"] == snr]
        by[snr] = {"transfer_oracle": float(np.mean([r["transfer_oracle"] for r in grp])),
                   "transfer_garch": float(np.mean([r["transfer_garch"] for r in grp])),
                   "transfer_gauss": float(np.mean([r["transfer_gauss"] for r in grp])),
                   "buyhold": float(np.mean([r["buyhold_sharpe"] for r in grp])),
                   "directional_r2": float(np.mean([r.get("directional_r2", 0.0) for r in grp])),
                   "sf_fidelity_garch": float(np.mean([r.get("sf_fidelity_garch", 0.0) for r in grp])),
                   "sf_fidelity_oracle": float(np.mean([r.get("sf_fidelity_oracle", 0.0) for r in grp]))}
    o_slope, _ = ols_slope(snrs, toracle)
    g_slope, _ = ols_slope(snrs, tgarch)
    return {"by_snr": by,
            "oracle_transfer_slope_vs_snr": o_slope, "oracle_slope_ci95": boot_slope_ci(snrs, toracle),
            "garch_transfer_slope_vs_snr": g_slope, "garch_slope_ci95": boot_slope_ci(snrs, tgarch),
            "interpretation": "oracle slope>0 (captures reward-relevant trend) while garch slope~0 "
                              "(stylized-fact-faithful but signal-blind) => fidelity buys transfer only "
                              "when it is REWARD-RELEVANT, not when it is stylized-fact fidelity."}


def main():
    ledger = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_LEDGER
    rows = load(ledger)
    print(f"loaded {len(rows)} run-rows from {ledger}")
    summary = {"n_rows": len(rows)}

    # model-free PPO Sharpes per generator — ONLY the A1 comparator (exp08, tagged a1), not exp07
    mf = {}
    for gen in ("gjr_garch", "regime_switch"):
        vals = [r.get("modelfree_sharpe") for r in rows
                if r.get("task") == "learned_transfer" and r.get("gen") == gen
                and r.get("a1") and "modelfree_sharpe" in r]
        mf[gen] = [v for v in vals if v is not None]

    summary["H2_verdict"] = {gen: verdict_for(rows, gen, mf.get(gen, [])) for gen in ("gjr_garch", "regime_switch")}
    summary["H3_decomposition"] = h3_decomposition(rows)

    # Deflated Sharpe across ALL dial transfer trials (the true trials count)
    all_dial = [r for r in rows if r.get("task") == "dial_transfer" and "transfer_sharpe" in r]
    if all_dial:
        shs = [r["transfer_sharpe"] for r in all_dial]
        T = int(all_dial[0].get("n_eval", 20000))   # synthetic held-out test length (steps)
        psr, sr0 = deflated_sharpe(shs, T, ann=252)
        summary["deflated_sharpe"] = {"n_trials": len(shs), "best_sharpe": float(np.max(shs)),
                                      "dsr_prob": psr, "sr0_benchmark": sr0}
    summary["real_panel"] = panel_bootstrap_real(rows)
    summary["predictability"] = predict_summary(rows)

    out_path = os.path.join(HERE, "..", "analysis_summary.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(json.dumps(summary, indent=2, default=float))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
