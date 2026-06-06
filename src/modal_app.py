"""
Modal experiment runner for: "Does simulator fidelity buy transfer?"
Dream-to-real RL in markets — leakage-controlled diagnostic.

One file, dispatched by config["task"]:
  - "smoke"           : tiny end-to-end sanity run
  - "synth_fidelity"  : generate an Oracle SDE market; fit {gaussian,garch,rnn_mdn,transformer_mdn};
                        held-out NLL + stylised-facts + fidelity vs Oracle truth  (H1)
  - "dial_transfer"   : parametric fidelity-dial simulator at level lambda -> DreamEnv -> PPO ->
                        transfer on Oracle held-out; returns (fidelity, transfer, gap)  (H2 headline)
  - "learned_transfer": fit a learned world model -> dream bank -> PPO -> transfer; also model-free
                        PPO on real train; baselines  (A1 / H3-bias / A2 learning-curve)
  - "real_fidelity"   : pooled yfinance equities -> shared-backbone world models -> fidelity (real H1)
  - "real_transfer"   : real DreamEnv+PPO vs model-free vs B&H/Random on held-out tickers+future (H2/H4)

Contract (per template): read everything from `config`; write progress.json + metrics.json into
RUN_DIR on the Volume; vol.commit(); return the metrics dict. The local entrypoint prints
RESULT_JSON:{...}. A "batch" key runs train.map over a list of configs (Modal fan-out).

NO-LOOKAHEAD CONTRACT (enforced in TradingEnv): the agent observes returns through index t-1 only;
the position chosen at step t earns r_t (the t-1 -> t return); reward = pos*r_t - cost*|dpos|.
All normalization stats are computed on the TRAIN split only.
"""
import json
import math
import os
import time

import numpy as np  # local env has numpy; np is only *called* inside @app.function bodies (remote, pinned 1.26.4)
import modal

SLUG = "dream-to-real-trading"
APP_NAME = f"research-{SLUG}"
VOL_NAME = f"research-{SLUG}"
DATA_ROOT = "/vol"

app = modal.App(APP_NAME)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "numpy==1.26.4",
        "pandas==2.2.2",
        "scipy==1.13.1",
        "torch==2.3.1",
        "stable-baselines3==2.3.2",
        "gymnasium==0.29.1",
        "arch==7.0.0",
        "statsmodels==0.14.2",
        "yfinance==0.2.40",
        "pyarrow==16.1.0",   # pandas to_parquet engine for the real-arm yfinance cache
        "tqdm",
    )
)

vol = modal.Volume.from_name(VOL_NAME, create_if_missing=True)


def _write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=float)


# =============================================================================
# 1. SYNTHETIC MARKETS (Oracle data-generating processes)
#    Each returns an array of log-returns. Parameters are KNOWN (the Oracle).
# =============================================================================
def _student_t_innov(n, nu, rng):
    """Standardized Student-t innovations (unit variance) for nu>2."""
    if nu is None or nu == float("inf") or nu > 200:
        return rng.standard_normal(n)
    z = rng.standard_t(nu, size=n)
    return z / math.sqrt(nu / (nu - 2.0))  # rescale to unit variance


def gen_gbm(n, rng, mu=0.0003, sigma=0.01, nu=None):
    """Geometric Brownian Motion log-returns: iid, no clustering, no leverage."""
    eps = _student_t_innov(n, nu, rng)
    return mu + sigma * eps


def gen_gjr_garch(n, rng, omega=2e-6, alpha=0.04, gamma=0.12, beta=0.88,
                  nu=5.0, mu=0.0002, burn=500):
    """GJR-GARCH(1,1)-t: fat tails (nu), clustering (alpha+beta+gamma/2), leverage (gamma>0)."""
    n_tot = n + burn
    eps = _student_t_innov(n_tot, nu, rng)
    r = np.zeros(n_tot)
    sig2 = np.zeros(n_tot)
    sig2[0] = omega / max(1e-12, (1 - alpha - beta - gamma / 2))
    for t in range(1, n_tot):
        prev = r[t - 1] - mu
        lev = 1.0 if prev < 0 else 0.0
        sig2[t] = omega + (alpha + gamma * lev) * prev * prev + beta * sig2[t - 1]
        r[t] = mu + math.sqrt(sig2[t]) * eps[t]
    return r[burn:]


def gen_regime_switch(n, rng, p_stay=0.985, nu=5.0, mu=0.0002,
                      omega=2e-6, alpha=0.04, gamma=0.12,
                      beta_lo=0.83, beta_hi=0.88, vmult_hi=4.0, burn=500):
    """2-state Markov-switching GJR-GARCH: non-stationary, strictly richer than a single GARCH.
    Low-vol regime vs high-vol regime (higher persistence + variance multiplier).
    Persistence kept < 1 in BOTH states (alpha+beta+gamma/2 <= 0.05+0.90+0.04 = 0.99) so the
    conditional variance never explodes; non-stationarity comes from regime switching + vmult_hi."""
    n_tot = n + burn
    eps = _student_t_innov(n_tot, nu, rng)
    r = np.zeros(n_tot)
    sig2 = np.zeros(n_tot)
    state = np.zeros(n_tot, dtype=int)
    sig2[0] = omega / max(1e-12, (1 - alpha - beta_lo - gamma / 2))
    for t in range(1, n_tot):
        # Markov transition
        if rng.random() > p_stay:
            state[t] = 1 - state[t - 1]
        else:
            state[t] = state[t - 1]
        beta = beta_hi if state[t] == 1 else beta_lo
        vm = vmult_hi if state[t] == 1 else 1.0
        prev = r[t - 1] - mu
        lev = 1.0 if prev < 0 else 0.0
        sig2[t] = vm * omega + (alpha + gamma * lev) * prev * prev + beta * sig2[t - 1]
        sig2[t] = min(sig2[t], 1.0)  # safety guard against any pathological explosion (std<=100%)
        r[t] = mu + math.sqrt(sig2[t]) * eps[t]
    return r[burn:]


def gen_momentum(n, rng, snr=0.5, psi=0.96, base_sigma=0.01, nu=5.0, mu=0.0002,
                 omega=2e-6, alpha=0.04, gamma=0.12, beta=0.88, burn=500):
    """A LEARNABLE-EDGE market: a hidden persistent trend s_t (AR(1), persistence psi) added to the
    conditional mean, on top of GJR-GARCH vol (so it still carries the stylized facts).
        s_t = psi*s_{t-1} + sigma_s*xi_t ;  r_t = mu + s_t + sqrt(sig2_t)*eps_t
    Because s_t is slowly-varying, E[r_t | recent returns] ~ s_t is INFERABLE from a return window,
    giving a genuine, transferable DIRECTIONAL edge. `snr` = Var(trend)/base variance controls its
    strength; snr=0 => no edge (pure GJR-GARCH, like the exp04 case). A GARCH fit to this data
    reproduces the vol/tails (high stylized-fact fidelity) but CANNOT represent the trend (it is
    signal-blind) — the basis for the reward-relevant-vs-stylized-fact contrast."""
    n_tot = n + burn
    eps = _student_t_innov(n_tot, nu, rng)
    sig2 = np.zeros(n_tot); r = np.zeros(n_tot); s = np.zeros(n_tot)
    sigma_s = base_sigma * math.sqrt(max(0.0, snr) * (1 - psi * psi))
    sig2[0] = omega / max(1e-9, (1 - alpha - beta - gamma / 2))
    for t in range(1, n_tot):
        s[t] = psi * s[t - 1] + sigma_s * rng.standard_normal()
        prev = r[t - 1] - mu - s[t - 1]
        lev = 1.0 if prev < 0 else 0.0
        sig2[t] = omega + (alpha + gamma * lev) * prev * prev + beta * sig2[t - 1]
        sig2[t] = min(sig2[t], 1.0)
        r[t] = mu + s[t] + math.sqrt(sig2[t]) * eps[t]
    return r[burn:]


ORACLES = {"gbm": gen_gbm, "gjr_garch": gen_gjr_garch, "regime_switch": gen_regime_switch}

# Default Oracle parameters per generator (the KNOWN truth).
ORACLE_PARAMS = {
    "gbm": dict(mu=0.0003, sigma=0.01, nu=None),
    "gjr_garch": dict(omega=2e-6, alpha=0.04, gamma=0.12, beta=0.88, nu=5.0, mu=0.0002),
    "regime_switch": dict(p_stay=0.985, nu=5.0, mu=0.0002, omega=2e-6, alpha=0.04,
                          gamma=0.12, beta_lo=0.83, beta_hi=0.88, vmult_hi=4.0),
}


def oracle_returns(gen, n, seed, **override):
    rng = np.random.default_rng(seed)
    params = dict(ORACLE_PARAMS[gen])
    params.update(override)
    return ORACLES[gen](n, rng, **params)


_ORACLE_STD = {}


def _oracle_std(gen):
    """Cached unconditional std of the Oracle process — the constant vol target for the dial."""
    if gen not in _ORACLE_STD:
        _ORACLE_STD[gen] = float(np.std(oracle_returns(gen, 200_000, seed=12345)))
    return _ORACLE_STD[gen]


def momentum_returns(snr, n, seed):
    return gen_momentum(n, np.random.default_rng(seed), snr=snr)


def garch_sim_returns(train_r, n, seed=0):
    """Fit GJR-GARCH-t to TRAIN returns and SIMULATE n returns: reproduces vol-clustering + fat tails
    (high stylized-fact fidelity) but has a CONSTANT conditional mean — i.e. SIGNAL-BLIND (no trend)."""
    from arch import arch_model
    scale = 100.0
    am = arch_model(np.asarray(train_r) * scale, vol="GARCH", p=1, o=1, q=1, dist="t", mean="Constant")
    res = am.fit(disp="off", show_warning=False)
    sim = am.simulate(res.params, n)
    return np.asarray(sim["data"]) / scale


def directional_r2(returns, L):
    """Realizable directional edge: R^2 of next return regressed on the mean of the prior-L returns."""
    r = np.asarray(returns, float)
    if len(r) < L + 50:
        return 0.0
    x = np.array([r[i:i + L].mean() for i in range(len(r) - L)])
    y = r[L:]
    if x.std() < 1e-12:
        return 0.0
    c = np.corrcoef(x, y)[0, 1]
    return float(c * c)


def dial_returns(gen, lam, n, seed):
    """Fidelity dial: interpolate the Oracle parameters from Gaussian-degenerate (lam=0) to
    Oracle-matched (lam=1), holding the parametric class fixed. lam in [0,1].
    Tails: nu -> inf at lam=0 (Gaussian). Clustering/leverage: alpha,gamma,(beta) -> 0 at lam=0
    (constant variance). At lam=0 the simulator is Gaussian-IID; at lam=1 it equals the Oracle."""
    rng = np.random.default_rng(seed)
    p = dict(ORACLE_PARAMS[gen])
    if gen == "gbm":
        return ORACLES[gen](n, rng, **p)  # GBM has no dial knobs (light-tailed control)
    # interpolate tail: 1/nu scales with lam (nu_eff -> inf as lam->0)
    nu_oracle = p["nu"]
    inv_nu = lam * (1.0 / nu_oracle)
    nu_eff = (1.0 / inv_nu) if inv_nu > 1e-6 else None
    p["nu"] = nu_eff
    # interpolate clustering/leverage toward 0
    base_omega = p["omega"]
    if gen == "gjr_garch":
        a0, g0, b0 = p["alpha"], p["gamma"], p["beta"]
        p["alpha"] = lam * a0
        p["gamma"] = lam * g0
        p["beta"] = lam * b0
        # keep unconditional variance ~constant: omega = uvar*(1-a-b-g/2)
        uvar = base_omega / (1 - a0 - b0 - g0 / 2)
        p["omega"] = uvar * max(1e-6, (1 - p["alpha"] - p["beta"] - p["gamma"] / 2))
    elif gen == "regime_switch":
        a0, g0, bl0, bh0 = p["alpha"], p["gamma"], p["beta_lo"], p["beta_hi"]
        uvar = base_omega / (1 - a0 - bl0 - g0 / 2)  # low-regime unconditional variance (Oracle)
        p["alpha"] = lam * a0
        p["gamma"] = lam * g0
        p["beta_lo"] = lam * bl0
        p["beta_hi"] = lam * bh0
        p["vmult_hi"] = 1.0 + lam * (p["vmult_hi"] - 1.0)  # -> 1 (no regime) at lam=0
        # hold low-regime unconditional variance constant across lambda (deconfound vol level)
        p["omega"] = uvar * max(1e-9, (1 - p["alpha"] - p["beta_lo"] - p["gamma"] / 2))
    r = ORACLES[gen](n, rng, **p)
    # Final deconfounding: rescale to the Oracle's unconditional std so the dial varies only the
    # SHAPE of returns (tails/clustering/leverage — all scale-invariant), never the volatility level.
    s = float(r.std())
    if s > 1e-12:
        m = float(r.mean())
        r = m + (r - m) * (_oracle_std(gen) / s)
    return r


# =============================================================================
# 2. STYLISED FACTS + FIDELITY SCORE (Cont 2001 estimators)
# =============================================================================
def excess_kurtosis(r):
    r = np.asarray(r); m = r.mean(); s = r.std()
    if s < 1e-12:
        return 0.0
    return float(((r - m) ** 4).mean() / s ** 4 - 3.0)


def hill_tail_index(r, frac=0.05):
    """Hill estimator of the tail index alpha (both tails, |r|). Returns alpha (>0)."""
    x = np.sort(np.abs(np.asarray(r)))[::-1]
    k = max(10, int(frac * len(x)))
    k = min(k, len(x) - 1)
    xk = x[:k]
    xk = xk[xk > 0]
    if len(xk) < 5 or x[k] <= 1e-12:
        return float("nan")
    logs = np.log(xk) - math.log(x[k])
    hill = logs.mean()
    return float(1.0 / hill) if hill > 1e-12 else float("nan")


def acf(x, lags):
    x = np.asarray(x) - np.mean(x)
    denom = np.sum(x * x)
    out = []
    for L in lags:
        if L >= len(x) or denom < 1e-18:
            out.append(0.0)
        else:
            out.append(float(np.sum(x[L:] * x[:-L]) / denom))
    return np.array(out)


def vol_clustering(r, lags=range(1, 11)):
    """Mean ACF of |r| over lags (positive, slow decay = clustering)."""
    return float(np.mean(acf(np.abs(r), list(lags))))


def leverage_l1(r, k=1):
    """Leverage: corr(r_t, r^2_{t+k}); negative for equities."""
    r = np.asarray(r)
    a = r[:-k]
    b = r[k:] ** 2
    if a.std() < 1e-12 or b.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def raw_acf1(r):
    return float(acf(r, [1])[0])


def stylised_facts(r):
    return {
        "excess_kurtosis": excess_kurtosis(r),
        "tail_index": hill_tail_index(r),
        "vol_clustering": vol_clustering(r),
        "leverage": leverage_l1(r),
        "raw_acf1": raw_acf1(r),
        "agg_gauss": excess_kurtosis(np.add.reduceat(
            np.asarray(r)[: (len(r) // 5) * 5], np.arange(0, (len(r) // 5) * 5, 5))),
    }


def fidelity_score(sim_facts, target_facts):
    """exp(-mean normalized abs error) over the 4 discriminating facts. In (0,1], 1=perfect."""
    keys = ["excess_kurtosis", "tail_index", "vol_clustering", "leverage"]
    scales = {
        "excess_kurtosis": max(abs(target_facts["excess_kurtosis"]), 1.0),
        "tail_index": max(abs(target_facts["tail_index"]), 1.0),
        "vol_clustering": max(abs(target_facts["vol_clustering"]), 0.05),
        "leverage": max(abs(target_facts["leverage"]), 0.02),
    }
    errs = []
    for k in keys:
        sv, tv = sim_facts[k], target_facts[k]
        if sv is None or tv is None or (isinstance(sv, float) and math.isnan(sv)):
            errs.append(1.0)
        else:
            errs.append(min(abs(sv - tv) / scales[k], 3.0))  # cap each error at 3
    return float(math.exp(-float(np.mean(errs))))


# =============================================================================
# 3. LEARNED WORLD MODELS (MDN head; RNN and causal Transformer backbones)
# =============================================================================
def _make_windows(returns, L):
    """Context windows -> next return. X:(N,L), y:(N,). Causal: y[i]=r[i+L] from X=r[i:i+L]."""
    r = np.asarray(returns, dtype=np.float32)
    N = len(r) - L
    if N <= 0:
        return np.zeros((0, L), np.float32), np.zeros((0,), np.float32)
    X = np.stack([r[i:i + L] for i in range(N)]).astype(np.float32)
    y = r[L:L + N].astype(np.float32)
    return X, y


def _make_windows_segments(segments, L):
    """Window each segment SEPARATELY (no window straddles a segment/ticker boundary), then concat."""
    Xs, ys = [], []
    for seg in segments:
        X, y = _make_windows(seg, L)
        if len(X):
            Xs.append(X); ys.append(y)
    if not Xs:
        return np.zeros((0, L), np.float32), np.zeros((0,), np.float32)
    return np.concatenate(Xs), np.concatenate(ys)


def build_world_model(kind, L, K, device):
    import torch
    import torch.nn as nn

    class MDNHead(nn.Module):
        def __init__(self, d_in, K):
            super().__init__()
            self.pi = nn.Linear(d_in, K)
            self.mu = nn.Linear(d_in, K)
            self.sig = nn.Linear(d_in, K)

        def forward(self, h):
            import torch.nn.functional as F
            pi = F.log_softmax(self.pi(h), dim=-1)
            mu = self.mu(h)
            sig = F.elu(self.sig(h)) + 1.0 + 1e-4  # ELU+1 -> positive
            return pi, mu, sig

    class RNNMDN(nn.Module):
        def __init__(self, L, K, hidden=64):
            super().__init__()
            self.lstm = nn.LSTM(input_size=1, hidden_size=hidden, batch_first=True)
            self.head = MDNHead(hidden, K)

        def forward(self, x):  # x:(B,L)
            out, _ = self.lstm(x.unsqueeze(-1))
            return self.head(out[:, -1, :])

    class TransformerMDN(nn.Module):
        def __init__(self, L, K, d=64, nhead=4, layers=2):
            super().__init__()
            self.proj = nn.Linear(1, d)
            self.pos = nn.Parameter(torch.zeros(1, L, d))
            enc = nn.TransformerEncoderLayer(d_model=d, nhead=nhead, dim_feedforward=4 * d,
                                             batch_first=True, dropout=0.0)
            self.enc = nn.TransformerEncoder(enc, num_layers=layers)
            self.head = MDNHead(d, K)
            self.L = L

        def forward(self, x):  # x:(B,L)
            B, L = x.shape
            h = self.proj(x.unsqueeze(-1)) + self.pos[:, :L, :]
            mask = torch.triu(torch.ones(L, L, device=x.device) * float("-inf"), diagonal=1)
            h = self.enc(h, mask=mask)
            return self.head(h[:, -1, :])

    model = (RNNMDN(L, K) if kind == "rnn_mdn" else TransformerMDN(L, K)).to(device)
    return model


def mdn_nll(pi_logp, mu, sig, y, l2_pi=0.1):
    import torch
    y = y.unsqueeze(-1)
    comp = -0.5 * math.log(2 * math.pi) - torch.log(sig) - 0.5 * ((y - mu) / sig) ** 2
    ll = torch.logsumexp(pi_logp + comp, dim=-1)
    nll = -ll.mean()
    pi = pi_logp.exp()
    reg = l2_pi * (pi ** 2).sum(dim=-1).mean()  # discourage component collapse
    return nll + reg, nll


def train_world_model(kind, train_r, val_r, L=60, K=5, epochs=20, lr=1e-3, batch=256,
                      seed=0, device=None, progress_cb=None):
    """Fit a learned MDN world model on TRAIN returns only. `train_r`/`val_r` may be a single 1-D
    array OR a list of segments (e.g. per-ticker) — segments are windowed separately so no window
    straddles a boundary. Normalization stats are computed on the TRAIN data only. Returns
    (model, held-out val NLL in RAW return units, (mu_n, sd_n))."""
    import torch
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)
    is_seg = isinstance(train_r, (list, tuple))
    tr_all = np.concatenate([np.asarray(s) for s in train_r]) if is_seg else np.asarray(train_r)
    mu_n, sd_n = float(np.mean(tr_all)), float(np.std(tr_all) + 1e-12)  # train-only normalization

    def _win(x):
        if isinstance(x, (list, tuple)):
            return _make_windows_segments([(np.asarray(s) - mu_n) / sd_n for s in x], L)
        return _make_windows((np.asarray(x) - mu_n) / sd_n, L)

    Xtr, ytr = _win(train_r)
    Xvl, yvl = _win(val_r)
    model = build_world_model(kind, L, K, device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    Xtr_t = torch.tensor(Xtr, device=device); ytr_t = torch.tensor(ytr, device=device)
    Xvl_t = torch.tensor(Xvl, device=device); yvl_t = torch.tensor(yvl, device=device)
    n = len(Xtr_t)
    best_val = float("inf"); best_state = None
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            pi, mu, sig = model(Xtr_t[idx])
            loss, _ = mdn_nll(pi, mu, sig, ytr_t[idx])
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():                        # BATCHED val pass (avoids OOM on the pooled real arm)
            vsum, vn = 0.0, 0
            for j in range(0, len(Xvl_t), batch):
                pi, mu, sig = model(Xvl_t[j:j + batch])
                _, vb = mdn_nll(pi, mu, sig, yvl_t[j:j + batch])
                bs = int(Xvl_t[j:j + batch].shape[0]); vsum += float(vb) * bs; vn += bs
            vnll = vsum / max(vn, 1)
        if vnll < best_val:                          # checkpoint best-val weights
            best_val = vnll
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if progress_cb:
            progress_cb(ep + 1, epochs, vnll)
    if best_state is not None:                       # restore best weights for sampling/eval
        model.load_state_dict(best_state)
    raw_val_nll = best_val + math.log(sd_n)          # normalized-unit NLL -> RAW return units (fair vs Gaussian/GARCH)
    return model, raw_val_nll, (mu_n, sd_n)


def sample_paths_from_model(model, seed_contexts, n_steps, mu_std, temperature=1.0,
                            n_paths=64, seed=0, device=None):
    """Autoregressively generate return paths (in RAW return units) from a learned MDN model.
    seed_contexts: (M,L) normalized real TRAIN windows to seed from. Returns (n_paths, n_steps) raw."""
    import torch
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    g = torch.Generator(device=device); g.manual_seed(seed)
    mu_n, sd_n = mu_std
    M, L = seed_contexts.shape
    idx = torch.randint(0, M, (n_paths,), generator=g, device=device)
    ctx = torch.tensor(seed_contexts[idx.cpu().numpy()], device=device)  # (n_paths,L) normalized
    out = torch.zeros(n_paths, n_steps, device=device)
    model.eval()
    with torch.no_grad():
        for t in range(n_steps):
            pi_logp, mu, sig = model(ctx)
            pi = pi_logp.exp()
            comp = torch.multinomial(pi, 1, generator=g).squeeze(-1)
            sel_mu = mu.gather(1, comp.unsqueeze(-1)).squeeze(-1)
            sel_sig = sig.gather(1, comp.unsqueeze(-1)).squeeze(-1) * temperature
            z = torch.randn(n_paths, generator=g, device=device)
            nxt = sel_mu + sel_sig * z  # normalized units
            out[:, t] = nxt
            ctx = torch.cat([ctx[:, 1:], nxt.unsqueeze(-1)], dim=1)
    raw = out.cpu().numpy() * sd_n + mu_n
    return raw


# =============================================================================
# 4. TRADING ENV (no-lookahead) + PPO + financial metrics
# =============================================================================
def make_trading_env(L, cost_bps, train_mu, train_sd):
    import gymnasium as gym
    from gymnasium import spaces

    class TradingEnv(gym.Env):
        """obs = normalized last L returns through t-1; action in {0:short,1:flat,2:long}.
        Position chosen from info<=t-1 earns r_t (the next return). reward = pos*r_t - cost*|dpos|.
        `paths` is a list of 1-D raw-return arrays; each episode walks one path."""
        metadata = {}

        def __init__(self, paths):
            super().__init__()
            self.paths = paths
            self.L = L
            self.cost = cost_bps * 1e-4
            self.observation_space = spaces.Box(-10.0, 10.0, (L,), dtype=np.float32)
            self.action_space = spaces.Discrete(3)
            self._rng = np.random.default_rng(0)

        def reset(self, seed=None, options=None):
            super().reset(seed=seed)
            if seed is not None:
                self._rng = np.random.default_rng(seed)
            self.path = self.paths[self._rng.integers(len(self.paths))]
            self.t = self.L          # first decision uses returns[0:L] (info through t-1=L-1)
            self.pos = 0
            return self._obs(), {}

        def _obs(self):
            w = self.path[self.t - self.L:self.t]
            return ((w - train_mu) / train_sd).astype(np.float32)

        def step(self, action):
            new_pos = action - 1                       # {0,1,2} -> {-1,0,+1}
            r_t = self.path[self.t]                     # the t-1 -> t return (unseen at decision)
            reward = new_pos * r_t - self.cost * abs(new_pos - self.pos)
            self.pos = new_pos
            self.t += 1
            terminated = self.t >= len(self.path)
            obs = self._obs() if not terminated else np.zeros(self.L, np.float32)
            return obs, float(reward), bool(terminated), False, {}

    return TradingEnv


def train_ppo(paths, L, cost_bps, train_mu, train_sd, total_timesteps=150_000, seed=0):
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    EnvCls = make_trading_env(L, cost_bps, train_mu, train_sd)
    venv = DummyVecEnv([lambda: EnvCls(paths)])
    venv.seed(seed)
    model = PPO("MlpPolicy", venv, verbose=0, seed=seed, device="cpu",
                n_steps=2048, batch_size=256, gamma=0.99, ent_coef=0.01,
                policy_kwargs=dict(net_arch=[64, 64]))
    model.learn(total_timesteps=total_timesteps)
    return model


def eval_policy_on_path(model, path, L, cost_bps, train_mu, train_sd):
    """Deterministic backtest of a trained policy on ONE raw-return path. Returns per-step strat returns."""
    cost = cost_bps * 1e-4
    pos = 0
    strat = []
    t = L
    while t < len(path):
        w = ((path[t - L:t] - train_mu) / train_sd).astype(np.float32)
        action, _ = model.predict(w, deterministic=True)
        new_pos = int(action) - 1
        r_t = path[t]
        strat.append(new_pos * r_t - cost * abs(new_pos - pos))
        pos = new_pos
        t += 1
    return np.array(strat)


def financial_metrics(strat, ann=252):
    strat = np.asarray(strat, dtype=float)
    if len(strat) < 2:
        return dict(sharpe=0.0, cum_return=0.0, max_drawdown=0.0, calmar=0.0, win_rate=0.0)
    sd = float(strat.std())
    sharpe = float(strat.mean() / sd * math.sqrt(ann)) if sd > 1e-12 else 0.0
    eq = np.exp(np.cumsum(strat))                       # log-return-consistent compounding
    cum = float(eq[-1] - 1.0)
    peak = np.maximum.accumulate(eq)
    mdd = float(np.max((peak - eq) / peak)) if len(eq) else 0.0
    ann_ret = float(np.exp(strat.mean() * ann) - 1.0)   # annualized from mean log-return
    calmar = float(ann_ret / mdd) if mdd > 1e-9 else 0.0
    win = float(np.mean(strat > 0))
    return dict(sharpe=sharpe, cum_return=cum, max_drawdown=mdd, calmar=calmar, win_rate=win)


def block_bootstrap_sharpe_ci(strat, n_boot=500, block=20, ann=252, seed=0):
    rng = np.random.default_rng(seed)
    strat = np.asarray(strat); n = len(strat)
    if n < block + 1:
        return (0.0, 0.0)
    sh = []
    nblocks = int(np.ceil(n / block))
    for _ in range(n_boot):
        starts = rng.integers(0, n - block + 1, size=nblocks)
        samp = np.concatenate([strat[s:s + block] for s in starts])[:n]
        if samp.std() > 1e-12:
            sh.append(samp.mean() / samp.std() * math.sqrt(ann))
    if not sh:
        return (0.0, 0.0)
    return (float(np.percentile(sh, 2.5)), float(np.percentile(sh, 97.5)))


def buy_and_hold(path, L, ann=252):
    return financial_metrics(np.asarray(path[L:]), ann)


def momentum_policy(path, L, cost_bps, ann=252):
    """No-lookahead time-series momentum: go long/short on the sign of the mean of the last L
    returns (info through t-1), earn r_t. A trivial non-RL directional baseline."""
    cost = cost_bps * 1e-4
    pos = 0; strat = []
    for t in range(L, len(path)):
        sig = float(np.mean(path[t - L:t]))
        new_pos = 1 if sig > 0 else (-1 if sig < 0 else 0)
        strat.append(new_pos * path[t] - cost * abs(new_pos - pos)); pos = new_pos
    return financial_metrics(np.array(strat), ann)


def random_policy(path, L, cost_bps, seed=0, ann=252):
    rng = np.random.default_rng(seed)
    cost = cost_bps * 1e-4
    pos = 0; strat = []
    for t in range(L, len(path)):
        new_pos = int(rng.integers(-1, 2))
        strat.append(new_pos * path[t] - cost * abs(new_pos - pos)); pos = new_pos
    return financial_metrics(np.array(strat), ann)


# =============================================================================
# 5. TASK DISPATCH
# =============================================================================
def _classical_fidelity(gen, train_r, val_r, target_facts, seed):
    """Gaussian-IID and GARCH(1,1) fit on TRAIN; held-out NLL on val; stylised facts of a big sample."""
    out = {}
    # Gaussian-IID
    mu, sd = float(np.mean(train_r)), float(np.std(train_r) + 1e-12)
    gauss_nll = float(0.5 * math.log(2 * math.pi) + math.log(sd)
                      + np.mean(((np.asarray(val_r) - mu) / sd) ** 2) * 0.5)
    g_sim = np.random.default_rng(seed).normal(mu, sd, size=200_000)
    out["gaussian"] = dict(val_nll=gauss_nll, fidelity=fidelity_score(stylised_facts(g_sim), target_facts),
                           facts=stylised_facts(g_sim))
    # GARCH(1,1)-t via arch: fit on TRAIN, evaluate held-out 1-step NLL on VAL with FIXED params.
    try:
        from arch import arch_model
        scale = 100.0  # arch prefers ~percent returns
        am = arch_model(np.asarray(train_r) * scale, vol="GARCH", p=1, o=1, q=1, dist="t", mean="Constant")
        res = am.fit(disp="off", show_warning=False)
        # out-of-sample NLL: filter the fitted params over VAL (no refit), convert to RAW-return units.
        # Change of variables x=scale*r => -log f_r = -log f_x - log(scale).
        am_val = arch_model(np.asarray(val_r) * scale, vol="GARCH", p=1, o=1, q=1, dist="t", mean="Constant")
        res_val = am_val.fix(res.params)
        garch_nll = float(-res_val.loglikelihood / res_val.nobs - math.log(scale))
        simd = am.simulate(res.params, 200_000)
        g_facts = stylised_facts(np.asarray(simd["data"]) / scale)
        out["garch"] = dict(val_nll=garch_nll, fidelity=fidelity_score(g_facts, target_facts), facts=g_facts)
    except Exception as e:
        out["garch"] = dict(val_nll=None, fidelity=None, facts=None, error=str(e)[:200])
    return out


@app.function(image=image, volumes={DATA_ROOT: vol}, gpu="L4", timeout=6 * 60 * 60)
def train(config: dict) -> dict:
    import torch
    exp_id = config.get("exp_id", "exp")
    task = config.get("task", "smoke")
    run_dir = os.path.join(DATA_ROOT, exp_id)
    os.makedirs(run_dir, exist_ok=True)
    progress_path = os.path.join(run_dir, "progress.json")
    metrics_path = os.path.join(run_dir, "metrics.json")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    def prog(**kw):
        _write_json(progress_path, {"exp_id": exp_id, "task": task, "done": False, **kw})
        vol.commit()

    seed = int(config.get("seed", 0))
    L = int(config.get("L", 60))
    K = int(config.get("K", 5))
    cost_bps = float(config.get("cost_bps", 5.0))
    gen = config.get("gen", "gjr_garch")
    ann = int(config.get("ann", 252))
    n_train = int(config.get("n_train", 60_000))
    n_eval = int(config.get("n_eval", 20_000))
    ppo_steps = int(config.get("ppo_steps", 150_000))
    if bool(config.get("smoke", False)):
        n_train, n_eval, ppo_steps = 4_000, 2_000, 4_000

    prog(stage="start")
    t0 = time.time()

    # Oracle "real" data: train/val/test splits (chronological, never shuffled).
    real_all = oracle_returns(gen, n_train + 2 * n_eval, seed=1000 + seed)
    tr = real_all[:n_train]
    vl = real_all[n_train:n_train + n_eval]
    te = real_all[n_train + n_eval:]
    train_mu, train_sd = float(np.mean(tr)), float(np.std(tr) + 1e-12)
    # Oracle ground-truth stylised facts (huge independent sample) = the fidelity TARGET.
    target_facts = stylised_facts(oracle_returns(gen, 500_000, seed=777))

    metrics = {"exp_id": exp_id, "task": task, "gen": gen, "seed": seed, "smoke": bool(config.get("smoke"))}

    if task in ("smoke", "synth_fidelity"):
        # H1: fidelity of classical + learned simulators on held-out data.
        res = _classical_fidelity(gen, tr, vl, target_facts, seed)
        for kind in (["transformer_mdn"] if task == "smoke" else ["rnn_mdn", "transformer_mdn"]):
            model, vnll, mu_std = train_world_model(
                kind, tr, vl, L=L, K=K, epochs=(2 if config.get("smoke") else 20),
                seed=seed, device=device,
                progress_cb=lambda e, E, v: prog(stage=f"fit_{kind}", epoch=e, total=E, val_nll=v))
            ctx = _make_windows((np.asarray(tr) - mu_std[0]) / mu_std[1], L)[0]
            sim = sample_paths_from_model(model, ctx, n_steps=2000, mu_std=mu_std,
                                          n_paths=100, seed=seed, device=device).reshape(-1)
            res[kind] = dict(val_nll=float(vnll), fidelity=fidelity_score(stylised_facts(sim), target_facts),
                             facts=stylised_facts(sim))
        metrics["target_facts"] = target_facts
        metrics["simulators"] = res

    elif task == "dial_transfer":
        # H2 headline: parametric dial simulator at level lambda -> PPO -> transfer on Oracle test.
        lam = float(config["lam"])
        dial = dial_returns(gen, lam, n_train, seed=2000 + seed)
        dpaths = [dial]  # one long dream path (episodes walk slices)
        sim_big = dial_returns(gen, lam, 200_000, seed=999)
        fid = fidelity_score(stylised_facts(sim_big), target_facts)
        prog(stage="ppo", lam=lam, fidelity=fid)
        ppo = train_ppo(dpaths, L, cost_bps, train_mu, train_sd, total_timesteps=ppo_steps, seed=seed)
        # transfer = realized on Oracle held-out test; in-dream = on fresh dial path
        strat_real = eval_policy_on_path(ppo, te, L, cost_bps, train_mu, train_sd)
        strat_dream = eval_policy_on_path(ppo, dial_returns(gen, lam, n_eval, seed=3000 + seed),
                                          L, cost_bps, train_mu, train_sd)
        fm = financial_metrics(strat_real, ann)
        ci = block_bootstrap_sharpe_ci(strat_real, seed=seed)
        in_dream = financial_metrics(strat_dream, ann)["sharpe"]
        # reference points on the SAME held-out test (self-contained headline)
        bh = buy_and_hold(te, L, ann); rnd = random_policy(te, L, cost_bps, seed=seed, ann=ann)
        metrics.update(dict(lam=lam, fidelity=fid, cost_bps=cost_bps, n_eval=n_eval,
                            headline=bool(config.get("headline", False)),  # True only for exp04/exp05 dial
                            transfer_sharpe=fm["sharpe"], transfer_ci=ci, in_dream_sharpe=in_dream,
                            gap=in_dream - fm["sharpe"], buyhold_sharpe=bh["sharpe"],
                            random_sharpe=rnd["sharpe"], **{f"real_{k}": v for k, v in fm.items()}))

    elif task == "learned_transfer":
        # A1 / H3-bias / A2: fit a learned world model, dream-train PPO, compare to model-free PPO.
        kind = config.get("kind", "transformer_mdn")
        n_bars = int(config.get("n_bars", n_train))   # for learning curve (A5)
        tr_lc = tr[:n_bars]
        model, vnll, mu_std = train_world_model(
            kind, tr_lc, vl, L=L, K=K, epochs=(2 if config.get("smoke") else 20),
            seed=seed, device=device,
            progress_cb=lambda e, E, v: prog(stage=f"fit_{kind}", epoch=e, total=E, val_nll=v))
        ctx = _make_windows((np.asarray(tr_lc) - mu_std[0]) / mu_std[1], L)[0]
        bank = sample_paths_from_model(model, ctx, n_steps=1000, mu_std=mu_std, n_paths=200,
                                       temperature=float(config.get("temperature", 1.0)),
                                       seed=seed, device=device)
        dream_paths = [bank[i] for i in range(bank.shape[0])]
        fid = fidelity_score(stylised_facts(bank.reshape(-1)), target_facts)
        prog(stage="ppo_dream", kind=kind, fidelity=fid)
        ppo_dream = train_ppo(dream_paths, L, cost_bps, train_mu, train_sd, total_timesteps=ppo_steps, seed=seed)
        sd_real = eval_policy_on_path(ppo_dream, te, L, cost_bps, train_mu, train_sd)
        dream_fm = financial_metrics(sd_real, ann)
        # model-free PPO trained directly on the real TRAIN series (no world model)
        prog(stage="ppo_modelfree")
        ppo_mf = train_ppo([tr], L, cost_bps, train_mu, train_sd, total_timesteps=ppo_steps, seed=seed)
        mf_real = financial_metrics(eval_policy_on_path(ppo_mf, te, L, cost_bps, train_mu, train_sd), ann)
        # Oracle-dream (train PPO on TRUE-process paths) = world-model-bias reference
        oracle_paths = [oracle_returns(gen, n_train, seed=4000 + seed + i) for i in range(5)]
        ppo_oracle = train_ppo(oracle_paths, L, cost_bps, train_mu, train_sd, total_timesteps=ppo_steps, seed=seed)
        oracle_real = financial_metrics(eval_policy_on_path(ppo_oracle, te, L, cost_bps, train_mu, train_sd), ann)
        bh = buy_and_hold(te, L, ann); rnd = random_policy(te, L, cost_bps, seed=seed, ann=ann)
        metrics.update(dict(kind=kind, n_bars=n_bars, val_nll=float(vnll), fidelity=fid,
                            a1=bool(config.get("a1", False)),  # True only for exp08 (the A1 model-free comparator)
                            dream_sharpe=dream_fm["sharpe"], modelfree_sharpe=mf_real["sharpe"],
                            oracle_dream_sharpe=oracle_real["sharpe"],
                            world_model_bias=oracle_real["sharpe"] - dream_fm["sharpe"],
                            buyhold_sharpe=bh["sharpe"], random_sharpe=rnd["sharpe"],
                            dream_full=dream_fm, modelfree_full=mf_real))

    elif task == "predict_dial":
        # Reward-relevant vs stylized-fact fidelity. A market WITH a learnable directional trend (snr).
        # Compare a dream that CAPTURES the trend (Oracle) vs one that is stylized-fact-faithful but
        # SIGNAL-BLIND (GARCH) vs a Gaussian floor — all evaluated on the same held-out real test.
        snr = float(config["snr"])
        real_all = momentum_returns(snr, n_train + 2 * n_eval, seed=1000 + seed)
        trm, tem = real_all[:n_train], real_all[n_train + n_eval:]
        tmu, tsd = float(np.mean(trm)), float(np.std(trm) + 1e-12)
        tgt = stylised_facts(momentum_returns(snr, 300_000, seed=777))
        prog(stage="predict", snr=snr)
        # (1) ORACLE dream — captures the reward-relevant trend
        ppo_o = train_ppo([momentum_returns(snr, n_train, seed=2000 + seed)], L, cost_bps, tmu, tsd,
                          total_timesteps=ppo_steps, seed=seed)
        so = eval_policy_on_path(ppo_o, tem, L, cost_bps, tmu, tsd); t_oracle = financial_metrics(so, ann)
        # (2) GARCH dream — high stylized-fact fidelity but SIGNAL-BLIND (constant conditional mean)
        ppo_g = train_ppo([garch_sim_returns(trm, n_train, seed=seed)], L, cost_bps, tmu, tsd,
                          total_timesteps=ppo_steps, seed=seed)
        t_garch = financial_metrics(eval_policy_on_path(ppo_g, tem, L, cost_bps, tmu, tsd), ann)
        # (3) Gaussian floor
        ppo_n = train_ppo([np.random.default_rng(seed).normal(tmu, tsd, n_train)], L, cost_bps, tmu, tsd,
                          total_timesteps=ppo_steps, seed=seed)
        t_gauss = financial_metrics(eval_policy_on_path(ppo_n, tem, L, cost_bps, tmu, tsd), ann)
        bh = buy_and_hold(tem, L, ann); rnd = random_policy(tem, L, cost_bps, seed=seed, ann=ann)
        metrics.update(dict(
            gen="momentum", snr=snr, directional_r2=directional_r2(tem, L),
            transfer_oracle=t_oracle["sharpe"], transfer_garch=t_garch["sharpe"], transfer_gauss=t_gauss["sharpe"],
            oracle_ci=block_bootstrap_sharpe_ci(so, seed=seed),
            buyhold_sharpe=bh["sharpe"], random_sharpe=rnd["sharpe"],
            sf_fidelity_oracle=fidelity_score(stylised_facts(momentum_returns(snr, 200_000, seed=999)), tgt),
            sf_fidelity_garch=fidelity_score(stylised_facts(garch_sim_returns(trm, 200_000, seed=43)), tgt)))

    elif task in ("real_fidelity", "real_transfer"):
        metrics.update(_real_arm(task, config, L, K, cost_bps, ann, seed, device, prog))

    else:
        metrics["error"] = f"unknown task {task}"

    metrics["runtime_sec"] = round(time.time() - t0, 1)
    _write_json(metrics_path, metrics)
    _write_json(progress_path, {**metrics, "done": True})
    vol.commit()
    return metrics


def _real_arm(task, config, L, K, cost_bps, ann, seed, device, prog):
    """External-validity arm: pooled yfinance equities -> shared-backbone world model -> transfer.
    Survivorship NOT controlled (current liquid universe) -> results labelled an optimistic bound."""
    import pandas as pd
    import io
    import urllib.request
    tickers = config.get("tickers") or ["AAPL", "MSFT", "SPY", "QQQ", "AMZN", "GOOGL", "JPM",
                                        "XOM", "JNJ", "PG", "KO", "WMT", "NVDA", "META", "V",
                                        "HD", "BAC", "DIS", "CSCO", "INTC", "PFE", "T", "CVX", "MRK"]
    cache = os.path.join(DATA_ROOT, "real_cache_stooq.parquet")
    if os.path.exists(cache):
        px = pd.read_parquet(cache)
    else:
        prog(stage="download", n_tickers=len(tickers))
        # Stooq free daily CSV via stdlib urllib (robust; no fragile yfinance/Yahoo API client).
        cols = {}
        for t in tickers:
            try:
                url = f"https://stooq.com/q/d/l/?s={t.lower()}.us&i=d"
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=40) as resp:
                    raw = resp.read().decode("utf-8", "ignore")
                df = pd.read_csv(io.StringIO(raw))
                if "Close" not in df.columns or len(df) < 200:
                    continue
                df["Date"] = pd.to_datetime(df["Date"])
                cols[t] = df.set_index("Date")["Close"]
            except Exception:
                continue
        px = pd.DataFrame(cols).sort_index().dropna(how="all")
        if px.shape[1] >= 5:
            px.to_parquet(cache); vol.commit()
    rets = np.log(px / px.shift(1)).replace([np.inf, -np.inf], np.nan).dropna(how="all")
    if rets.shape[1] < 5 or len(rets) < 500:
        return {"error": f"insufficient real data: {rets.shape[1]} tickers x {len(rets)} rows", "task": task}
    # chronological + cross-sectional split: hold out last 15% time AND a few tickers
    cols = list(rets.columns)
    hold_tickers = set(config.get("hold_tickers", cols[-3:]))
    train_cols = [c for c in cols if c not in hold_tickers]
    n_time = len(rets); cut = int(0.7 * n_time); cutv = int(0.85 * n_time)
    # shared-backbone TRAIN segments: train tickers, train time (NO held-out tickers, NO future).
    # passed as a LIST so train_world_model windows each ticker separately (no cross-ticker windows).
    train_series = [rets[c].iloc[:cut].dropna().values.astype(np.float32) for c in train_cols]
    train_series = [s for s in train_series if len(s) > L + 50]
    val_series = [rets[c].iloc[cut:cutv].dropna().values.astype(np.float32) for c in train_cols]
    val_series = [s for s in val_series if len(s) > L + 5]
    if len(train_series) < 3 or len(val_series) < 1:
        return {"error": f"empty segments: train_cols={len(train_cols)}, train_series={len(train_series)}", "task": task}
    tr_concat = np.concatenate(train_series)
    train_mu, train_sd = float(np.mean(tr_concat)), float(np.std(tr_concat) + 1e-12)
    model, vnll, mu_std = train_world_model("transformer_mdn", train_series, val_series, L=L, K=K,
                                            epochs=(2 if config.get("smoke") else 15), seed=seed,
                                            device=device, batch=512,
                                            progress_cb=lambda e, E, v: prog(stage="fit_real", epoch=e, total=E, val_nll=v))
    out = {"val_nll": float(vnll), "n_train_tickers": len(train_cols), "hold_tickers": list(hold_tickers),
           "survivorship": "NOT controlled (current liquid universe) — real-arm results are an optimistic bound"}
    # dream-bank seed contexts from TRAIN segments only (per-ticker windows, train-only normalization).
    # RANDOM-sample across the whole universe (not the first 5000 = first tickers only).
    all_ctx = _make_windows_segments([(s - mu_std[0]) / mu_std[1] for s in train_series], L)[0]
    _crng = np.random.default_rng(seed)
    sel = _crng.choice(len(all_ctx), size=min(5000, len(all_ctx)), replace=False)
    ctx = all_ctx[sel]
    if task == "real_fidelity":
        sim = sample_paths_from_model(model, ctx, n_steps=1000, mu_std=mu_std, n_paths=100,
                                      seed=seed, device=device).reshape(-1)
        # fidelity TARGET on TRAIN-time / TRAIN-ticker returns only (what the model could have seen)
        target = stylised_facts(tr_concat)
        out["target_facts"] = target
        out["sim_facts"] = stylised_facts(sim)
        out["fidelity"] = fidelity_score(stylised_facts(sim), target)
        return out
    # real_transfer: dream-train PPO; eval on held-out tickers' FUTURE (held-out time)
    bank = sample_paths_from_model(model, ctx, n_steps=1000, mu_std=mu_std, n_paths=200, seed=seed, device=device)
    ppo_dream = train_ppo([bank[i] for i in range(bank.shape[0])], L, cost_bps, train_mu, train_sd,
                          total_timesteps=(4000 if config.get("smoke") else 150_000), seed=seed)
    ppo_mf = train_ppo(train_series, L, cost_bps, train_mu, train_sd,
                       total_timesteps=(4000 if config.get("smoke") else 150_000), seed=seed)
    rows = []
    # evaluate transfer on the held-out FUTURE of the WHOLE universe (out-of-time for all tickers;
    # additionally out-of-ticker for the held-out tickers) so the real arm is pooled over ~80 equities.
    eval_cols = list(hold_tickers) + train_cols
    for c in eval_cols:
        fut_s = rets[c].iloc[cutv:].dropna()
        fut = fut_s.values.astype(np.float32)
        if len(fut) < L + 30:
            continue
        ds = eval_policy_on_path(ppo_dream, fut, L, cost_bps, train_mu, train_sd)
        ms = eval_policy_on_path(ppo_mf, fut, L, cost_bps, train_mu, train_sd)
        d = financial_metrics(ds, ann); m = financial_metrics(ms, ann); bh = buy_and_hold(fut, L, ann)
        mom = momentum_policy(fut, L, cost_bps, ann)  # trivial non-RL directional baseline
        dates = [str(x)[:10] for x in fut_s.index[L:L + len(ds)]]  # calendar dates aligned to the strat series
        rows.append(dict(ticker=str(c), held_out_ticker=(c in hold_tickers),
                         dream_sharpe=d["sharpe"], modelfree_sharpe=m["sharpe"], buyhold_sharpe=bh["sharpe"],
                         momentum_sharpe=mom["sharpe"],
                         dates=dates,  # for date-correct PANEL bootstrap alignment in analyze.py
                         dream_strat=[round(float(x), 6) for x in ds],
                         modelfree_strat=[round(float(x), 6) for x in ms]))
    out["per_ticker"] = rows
    out["dream_sharpe_mean"] = float(np.mean([r["dream_sharpe"] for r in rows])) if rows else 0.0
    out["modelfree_sharpe_mean"] = float(np.mean([r["modelfree_sharpe"] for r in rows])) if rows else 0.0
    out["buyhold_sharpe_mean"] = float(np.mean([r["buyhold_sharpe"] for r in rows])) if rows else 0.0
    out["momentum_sharpe_mean"] = float(np.mean([r["momentum_sharpe"] for r in rows])) if rows else 0.0
    return out


@app.local_entrypoint()
def main(config_json: str = "{}"):
    # support the "@path/to/file.json" convention (Modal's CLI does not expand it itself)
    if config_json.startswith("@"):
        with open(config_json[1:]) as f:
            config_json = f.read()
    config = json.loads(config_json)
    if "batch" in config:
        batch = config["batch"]
        results = list(train.map(batch))
        print("RESULT_JSON:" + json.dumps({"batch_results": results}))
    else:
        gpu = config.get("gpu", "L4")
        fn = train.with_options(gpu=gpu) if gpu != "L4" else train
        result = fn.remote(config)
        print("RESULT_JSON:" + json.dumps(result))
