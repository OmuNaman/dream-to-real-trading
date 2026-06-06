# Does Simulator Fidelity Buy Transfer? A Leakage-Controlled Diagnosis of Generative Market World Models for Reinforcement-Learning Trading

Model-based RL trains agents inside a learned simulator (a "dream"), and generative-market research works hard to make such simulators *realistic* — reproducing the stylized facts of returns (fat tails, volatility clustering, leverage). This work asks the unexamined question at the intersection of those two literatures: **does a market simulator's stylized-fact fidelity actually predict how well an agent trained inside it transfers to real data?** Using a leakage-controlled testbed (known-SDE Oracle markets, a dense capacity-matched *fidelity dial*, a no-lookahead trading environment, and a Transformer–MDN world model), we find that **stylized-fact fidelity is *decoupled* from dream-to-real transfer** — and that what actually transfers is *reward-relevant* (directional) structure, which stylized facts do not capture.

📄 **Paper:** [paper/paper.pdf](paper/paper.pdf) · 🌐 **Project page:** https://omunaman.github.io/dream-to-real-trading/ · 💻 reproducible on [Modal](https://modal.com)

## Summary
- **Problem.** Generative-market models optimize stylized-fact fidelity; RL-trading is model-free. Nobody has tested whether fidelity predicts the *transfer* of an agent trained inside such a simulator.
- **Approach.** A known-SDE Oracle (GBM, GJR-GARCH-t, regime-switching) lets us build a *fidelity dial* — a continuous, capacity-matched axis of simulator fidelity — and decompose the dream-to-real gap. A causal **Transformer mixture-density** world model is wrapped as a Gym `DreamEnv`; PPO is trained only in the dream and backtested on held-out real data through a **no-lookahead** `TradingEnv`.
- **Headline result.** Across the full resolved fidelity range, the transfer-vs-fidelity regression slope spans zero (**+0.20 [−0.37, 0.60]** on GJR-GARCH; **−0.09 [−0.47, 0.23]** on regime-switching) and the agent **never beats Buy-and-Hold**: fidelity is **decoupled** from transfer.
- **Mechanism.** A known-SDE decomposition shows the gap is *not* a world-model-quality problem (world-model bias ≈ 0; a perfect Oracle dream transfers no better). A controlled predictability experiment isolates the cause: when the simulator captures **reward-relevant** directional structure, transfer rises in lockstep (**slope +2.83 [2.16, 4.99]**), while a stylized-fact-faithful but **signal-blind GARCH dream stays flat (+0.48, n.s.)**. This confirms, in markets, that reward-relevant conditional accuracy — not marginal fidelity — is what buys transfer.
- **Honest external validity.** On real S&P large-caps (world model trained on 80 tickers, evaluated on the held-out future of the full universe), the dream (−0.66), model-free PPO (−0.19), and a trivial momentum baseline (+0.05) all lose to Buy-and-Hold (+0.63); a Deflated Sharpe over all trials is insignificant (1.4×10⁻⁹). No agent has a real edge — and we explain why.

## Key results
| Setting | dream / Oracle | comparator | verdict |
|---|---|---|---|
| H2 fidelity→transfer (GJR-GARCH) | slope **+0.20** [−0.37, 0.60] | spans 0; never beats B&H | **decoupled** |
| H2 fidelity→transfer (regime-switch) | slope **−0.09** [−0.47, 0.23] | spans 0; never beats B&H | **decoupled** |
| Mechanism: reward-relevant dial | Oracle slope **+2.83** [2.16, 4.99] | GARCH +0.48 [−0.28, 1.49] (n.s.) | reward-relevance drives transfer |
| World-model bias (H3) | Oracle-dream ≈ learned-dream | bias ≈ **−0.19 / −0.04** | gap is not WM quality |
| Real S&P (held-out) | dream **−0.66**, model-free **−0.19** | Buy-and-Hold **+0.63** | RL ≪ B&H after costs |

All numbers are reproducible from [`results/results.jsonl`](results/results.jsonl) via [`src/analyze.py`](src/analyze.py).

## Repository layout
- `src/` — training code (`modal_app.py`), the headline analysis (`analyze.py`), plotting (`make_plots.py`), config generation (`gen_configs.py`), a local sanity suite (`local_sanity.py`), the real-data fetcher (`fetch_real.py`), and experiment `configs/`.
- `results/` — append-only `results.jsonl` ledger, `experiments_log.md`, and `analysis_summary.json` (the computed verdicts).
- `paper/` — LaTeX source, final figures, and the compiled `paper.pdf`.

## Reproduce
```bash
pip install -r requirements.txt
modal setup
# generate configs
python src/gen_configs.py
# smoke test, then the headline synthetic experiments (fan out via Modal .map)
modal run src/modal_app.py --config-json "@src/configs/exp00.json"
modal run src/modal_app.py --config-json "@src/configs/exp04_dial_garch.json"
modal run src/modal_app.py --config-json "@src/configs/exp06_predict.json"
# compute the verdict + make the figures
python src/analyze.py
python src/make_plots.py
```
The real arm (`exp11`/`exp12`) reads a cached price matrix on the Modal volume; `src/fetch_real.py` builds it locally (Yahoo/Stooq are blocked from cloud IPs). `src/local_sanity.py` validates the generators/estimators/analysis without any GPU.

## Citation
```bibtex
@misc{kudale2026fidelity,
  title={Does Simulator Fidelity Buy Transfer? A Leakage-Controlled Diagnosis of Generative Market World Models for Reinforcement-Learning Trading},
  author={Kudale, Prit and Dwivedi, Naman and Dandekar, Raj and Dandekar, Rajat and Panat, Sreedath},
  year={2026},
  note={Vizuara AI Labs}
}
```

## Disclaimer
This is a research proof-of-concept on historical backtests and synthetic markets. It is **not** trading advice or a live-trading system. The real-arm results use a current liquid universe without point-in-time membership and are an optimistic bound.
