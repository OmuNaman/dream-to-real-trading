"""Generate experiment config batches for modal_app.py.
Writes one JSON file per experiment group into configs/. Each "batch" file holds a list of
single-run configs that modal_app's local entrypoint runs via train.map (Modal fan-out).

Usage (local): py -3.13 gen_configs.py
Then run e.g.:  py -3.13 -m modal run modal_app.py --config-json "@configs/exp04_dial_garch.json"
"""
import json
import os

HERE = os.path.dirname(__file__)
CFG = os.path.join(HERE, "configs")
os.makedirs(CFG, exist_ok=True)

SEEDS = [0, 1, 2]
LAMS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.65, 0.8, 0.9, 1.0]  # dense fidelity dial (10 pts)


def dump(name, obj):
    with open(os.path.join(CFG, name), "w") as f:
        json.dump(obj, f, indent=2)
    print("wrote", name)


# exp02 — H1 fidelity of all simulators, per generator x seed
dump("exp02_synth_fidelity.json", {"exp_id": "exp02", "batch": [
    {"exp_id": f"exp02_{gen}_s{s}", "task": "synth_fidelity", "gen": gen, "seed": s, "L": 60, "K": 5}
    for gen in ["gbm", "gjr_garch", "regime_switch"] for s in SEEDS]})

# exp04 — H2 headline: dial transfer on gjr_garch (10 lambdas x 3 seeds)
dump("exp04_dial_garch.json", {"exp_id": "exp04", "batch": [
    {"exp_id": f"exp04_l{int(l*100):03d}_s{s}", "task": "dial_transfer", "gen": "gjr_garch",
     "lam": l, "seed": s, "L": 60, "cost_bps": 5.0, "ppo_steps": 200000, "headline": True}
    for l in LAMS for s in SEEDS]})

# exp05 — H2 + A6: dial transfer on regime_switch (non-stationary)
dump("exp05_dial_regime.json", {"exp_id": "exp05", "batch": [
    {"exp_id": f"exp05_l{int(l*100):03d}_s{s}", "task": "dial_transfer", "gen": "regime_switch",
     "lam": l, "seed": s, "L": 60, "cost_bps": 5.0, "ppo_steps": 200000, "headline": True}
    for l in LAMS for s in SEEDS]})

# exp07 — H3 finite-data + A2/A5: learning curve, Transformer vs RNN, vs #bars
dump("exp07_learning_curve.json", {"exp_id": "exp07", "batch": [
    {"exp_id": f"exp07_{kind}_n{nb}_s{s}", "task": "learned_transfer", "gen": "gjr_garch",
     "kind": kind, "n_bars": nb, "seed": s, "L": 60, "cost_bps": 5.0, "ppo_steps": 150000}
    for kind in ["rnn_mdn", "transformer_mdn"] for nb in [2000, 5000, 10000, 30000, 60000]
    for s in SEEDS]})

# exp08 — A1: learned dream vs model-free vs oracle-dream vs baselines, per generator
dump("exp08_a1_modelfree.json", {"exp_id": "exp08", "batch": [
    {"exp_id": f"exp08_{gen}_s{s}", "task": "learned_transfer", "gen": gen, "kind": "transformer_mdn",
     "seed": s, "L": 60, "cost_bps": 5.0, "ppo_steps": 150000, "a1": True}
    for gen in ["gjr_garch", "regime_switch"] for s in SEEDS]})

# exp09 — A3: transaction-cost penalty on/off (dial at high fidelity lam=0.9)
dump("exp09_cost.json", {"exp_id": "exp09", "batch": [
    {"exp_id": f"exp09_c{int(c)}_s{s}", "task": "dial_transfer", "gen": "gjr_garch",
     "lam": 0.9, "seed": s, "L": 60, "cost_bps": c, "ppo_steps": 150000}
    for c in [0.0, 5.0, 20.0] for s in SEEDS]})

# exp06 — reward-relevant vs stylized-fact fidelity (predictability dial): a market WITH a learnable
# directional trend (snr). Oracle dream (captures trend) vs GARCH dream (stylized-fact-faithful but
# signal-blind) vs Gaussian floor, on the same held-out real test.
dump("exp06_predict.json", {"exp_id": "exp06", "batch": [
    {"exp_id": f"exp06_snr{int(snr*10):03d}_s{s}", "task": "predict_dial", "snr": snr,
     "seed": s, "L": 60, "cost_bps": 5.0, "ppo_steps": 200000}
    for snr in [0.0, 0.25, 0.5, 1.0, 2.0, 4.0] for s in SEEDS]})

# exp11/12 — real arm (external validity). Single A100 runs.
dump("exp11_real_fidelity.json", {"exp_id": "exp11", "task": "real_fidelity", "seed": 0,
                                  "L": 60, "K": 5, "gpu": "A100-80GB"})
dump("exp12_real_transfer.json", {"exp_id": "exp12", "task": "real_transfer", "seed": 0,
                                  "L": 60, "K": 5, "cost_bps": 5.0, "gpu": "A100-80GB"})

if __name__ == "__main__":
    print("configs written to", CFG)
