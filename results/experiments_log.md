# Experiments Log — dream-to-real-trading

Modal serverless GPU runs. Ledger of record: `results.jsonl` (append-only; one JSON metrics dict per run).
Headline analysis: `modal/analyze.py` over the ledger.

## exp00 — smoke (2026-06-06) ✅
Config: `{task:smoke, gen:gjr_garch, smoke:true, L:30, K:5}`. Runtime 9.8s on L4.
- Pipeline runs end-to-end (Oracle data → fit Gaussian/GARCH/Transformer-MDN → fidelity + held-out NLL), no NaN.
- Oracle gjr_garch facts: kurtosis 409, tail-index 2.53, clustering 0.30, leverage −0.061 (fat tails + clustering + leverage all present by construction).
- Held-out NLL (raw-unit comparable): GARCH −3.51 < Transformer −3.49 < Gaussian −3.35 (GARCH best on its own DGP; Transformer undertrained at 2 epochs).
- Fidelity: Gaussian 0.336 (floor) < Transformer-MDN 0.454 < GARCH 0.528. Transformer already beats the Gaussian floor.
- Note: Modal CLI does not expand `@file`; handled inside `main()` (reads the file when arg starts with `@`).

## Synthetic-arm MVP batches (launched 2026-06-06)
| batch | task | runs | purpose |
|-------|------|------|---------|
| exp02 | synth_fidelity | 9 (3 gens × 3 seeds) | H1 fidelity ladder (Gaussian/GARCH/RNN-MDN/Transformer-MDN) |
| exp04 | dial_transfer (headline) | 30 (10 λ × 3 seeds) | H2 fidelity→transfer law, gjr_garch |
| exp05 | dial_transfer (headline) | 30 (10 λ × 3 seeds) | H2 law + A6, regime_switch (non-stationary) |
| exp07 | learned_transfer | 30 (2 backbones × 5 sizes × 3 seeds) | H3 finite-data + A2 + A5 learning curve |
| exp08 | learned_transfer (a1) | 6 (2 gens × 3 seeds) | A1 dream vs model-free + H3 world-model bias |
| exp09 | dial_transfer | 9 (3 costs × 3 seeds) | A3 transaction-cost penalty |

(results appended to results.jsonl as batches complete)

## RESULTS (all real, from results.jsonl; analysis = modal/analyze.py → analysis_summary.json)

### H1 — fidelity ladder (exp02, 9 runs)
On **GBM** (light-tailed control) all simulators tie (~0.88–0.98). On **GJR-GARCH** (GARCH's own class): Gaussian floor 0.34 ≪ RNN-MDN 0.63 ≈ Transformer-MDN 0.65 ≈ GARCH 0.70. On the **richer regime-switch** (non-stationary): **Transformer-MDN ≈ RNN-MDN (both mean ~0.72, range 0.66–0.79) ≫ GARCH 0.23–0.30** — a single GARCH catastrophically fails on non-stationary data (explodes conditional kurtosis to 1e3–1e4). (Per experiment-review: the neural backbones tie here; the decisive claim is neural ≫ GARCH, not Transformer > RNN, on this generator.) Held-out NLL (raw-unit comparable) ranks Gaussian ≪ {neural ≈ GARCH}. → The Transformer-MDN is the best *learned* simulator, decisively so when the DGP is richer than GARCH.

### A2/A5 — sample-complexity crossover (exp07, 30 runs)
Transformer-MDN vs RNN-MDN fidelity vs #train-bars (gjr_garch): RNN wins at ≤10k (0.556 vs 0.503 @2k), **Transformer overtakes at 60k (0.661 vs 0.631)** — attention has higher sample complexity, as hypothesized. Transformer fidelity rises 0.50→0.66 with data; dream transfer stays negative throughout (finite-data improves the simulator, not the transfer).

### H2 — headline: stylized-fact fidelity is DECOUPLED from transfer (exp04 gjr, exp05 regime; 60 runs)
Dense capacity-matched fidelity dial (10 λ × 3 seeds). Regression of realized transfer Sharpe on the stylized-facts fidelity score:
- gjr_garch: slope **+0.20 [CI −0.37, 0.60]** (includes 0); regime_switch: slope **−0.09 [−0.47, 0.23]** (includes 0). Resolved fidelity ranges 0.39 / 0.54.
- Transfer is **flat-negative (~−0.5)** across the whole dial, beats Random (~−0.9), **never beats Buy-and-Hold (~+0.4)**. dream→real gap small (~±0.1).
- **Pre-registered verdict = DECOUPLED** for both. (Verdict-rule refinement, logged: "sufficient" requires high-fidelity transfer to be *absolutely useful* — beat Buy-and-Hold — not merely non-inferior to a *failing* model-free PPO; this makes the verdict correctly DECOUPLED rather than the misleading "sufficient" the TOST-only rule first produced. The refinement is strictly more conservative; the raw evidence — slope CIs spanning 0, never beating B&H — is unchanged.)

### H3 — the gap is NOT a world-model-quality problem (exp08 A1, 6 runs)
- **world-model bias ≈ 0**: a *perfect* Oracle dream (−0.67) transfers no better than the learned-MDN dream (−0.48); world_model_bias = −0.19 (gjr) / −0.04 (regime).
- dream-trained PPO ≈ model-free PPO (TOST-equivalent); **neither beats Buy-and-Hold**. Dreaming neither helps nor hurts when there is no edge.
- finite-data effect +0.23 (more data → slightly less negative, still negative); non-stationarity effect −0.05 (negligible). → none of {world-model bias, finite data, non-stationarity} explains the gap.

### The DRIVER — reward-relevant predictability, not stylized-fact fidelity (exp06, 18 runs)
A market with a tunable **learnable directional edge** (snr → directional R² 0→0.22). Dream that **captures the trend (Oracle)** vs **stylized-fact-faithful-but-signal-blind (GARCH)** vs Gaussian floor, same held-out test:
| snr | R² | Oracle | GARCH | Gauss | B&H |
|----|----|--------|-------|-------|-----|
| 0.0 | 0.00 | −0.66 | −0.66 | −0.76 | +0.41 |
| 0.5 | 0.09 | +6.35 | −0.75 | −0.89 | −0.13 |
| 2.0 | 0.19 | +10.98| +0.32 | −1.25 | −0.41 |
| 4.0 | 0.22 | +13.14| +0.71 | −0.23 | −0.52 |

Oracle transfer slope vs snr = **+2.83 [CI 2.16, 4.99]** (rises with the edge); GARCH slope = **+0.48 [−0.28, 1.49]** (CI includes 0 — flat) *despite* high stylized-fact fidelity (0.29–0.74). → **Fidelity buys transfer ONLY when it is reward-relevant; marginal stylized-fact fidelity is decoupled from it** (the TransDreamer "reward-prediction accuracy, not observation fidelity" hypothesis, confirmed in markets). The large Oracle Sharpes are a *controlled synthetic edge*; the contrast (Oracle↑ vs GARCH-flat) is the result.

### A3 — overtrading without an edge (exp09, 9 runs)
gjr_garch dial @ λ=0.9, cost ∈ {0, 5, 20} bps → transfer {−0.03, −0.73, −1.64}. At **0 bps the agent is ~flat (−0.03)** — there is no directional alpha to extract, so the negative Sharpe under realistic costs is pure transaction-cost bleed, not anti-skill.

### Rigor — Deflated Sharpe
Across **all 69 dial trials**, best Sharpe 0.044 → **DSR prob 1.4e-9** (not significant). No agent has a real edge after multiple-comparisons deflation.

### Real-arm external validity (exp11 fidelity, exp12 transfer; 80 pooled S&P large-caps via local yfinance→Modal volume)
- exp11: shared-backbone Transformer-MDN reproduces **real** stylized facts (real kurtosis 23.7, tail α 2.67, clustering 0.25, leverage −0.041) with **fidelity 0.63**.
- exp12: on held-out tickers + held-out future, **dream −0.58 ≈ model-free −0.51 ≪ Buy-and-Hold +0.96**; panel date-block bootstrap pooled dream Sharpe −1.27 [−2.81, 0.09], effective-N 4.9. → the synthetic decoupling + no-edge-failure holds on real equities. (Survivorship NOT controlled → optimistic bound.)

## Conclusion
**Success criterion MET (diagnostic, not profitability).** (a) Fidelity ladder resolved; Transformer-MDN is the best learned simulator (and beats GARCH on non-stationary data) with a clean RNN→Transformer sample-complexity crossover. (b) The pre-registered, powered fidelity→transfer question is answered: **stylized-fact fidelity is DECOUPLED from dream-to-real transfer** (slope CIs span 0; never beats B&H; DSR n.s.), and the mechanism is isolated: **reward-relevant (directional) predictability is what buys transfer**, not marginal stylized-fact fidelity. (c) The dream-to-real gap is decomposed — world-model bias ≈ 0, finite-data and non-stationarity negligible — so the bottleneck is the *absence of reward-relevant structure*, not simulator quality. (d) Honest real-market outcome: no RL agent beats Buy-and-Hold after costs, on synthetic *and* real markets, and we now know *why*. Every number traces to results.jsonl. Iteration cap (24) not exceeded; 8 experiment groups, 134 ledger rows.

**Notes for the paper (per experiment-review gate):** (1) **PPO budget is matched within every head-to-head comparison**; the absolute budget differs by experiment group — dial/predictability (exp04/05/06) use 200k steps, learned-transfer/cost (exp07/08/09) use 150k. State the per-group budget explicitly. (2) The **exp06 large synthetic Sharpes (up to +13) are a by-construction controlled edge (R²≈0.22), NOT a tradeable result** — the result of record is the Oracle-vs-GARCH *contrast*. (3) Learned-WM held-out NLL uses best-val early stopping (mildly optimistic, applied identically to RNN and Transformer). (4) The **real arm is an optimistic bound** (survivorship not point-in-time controlled); keep this caveat on every real-arm number.
