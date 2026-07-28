import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Common import (SELECTED, MODELS_DIR, load_model, evaluate_paired, eps_cum,
                    rollout_agent, rollout_bs, BSDelta, HedgingEnv,
                    K, T, r, sigma, BASE_EVAL_SEED)
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

OUT = os.path.dirname(os.path.abspath(__file__))
STRIKES = [80, 85, 90, 95, 100, 105, 110, 115, 120]

models = {reg: load_model(cfg["run_name"]) for reg, cfg in SELECTED.items()}

# Per-strike table (RQ3): PPO and BS-delta std, plus eps_cum, both regimes
rows = []
for K_eval in STRIKES:
    row = {"K": K_eval}
    for reg, cfg in SELECTED.items():
        model, vecnorm = models[reg]
        agent, bench = evaluate_paired(model, vecnorm, cfg["beta"], K_eval=K_eval)
        Pi_agent = np.array([res["Pi_T"] for res in agent])
        Pi_bench = np.array([res["Pi_T"] for res in bench])
        row[f"std_PPO_{reg}"] = round(Pi_agent.std(ddof=1), 4)
        row[f"std_BS_{reg}"]  = round(Pi_bench.std(ddof=1), 4)
        row[f"eps_cum_{reg}"] = round(eps_cum(agent).mean(), 4)
    rows.append(row)
    print(row)
pd.DataFrame(rows).to_csv(os.path.join(OUT, "strike_table.csv"), index=False)

# Policy vs Delta surfaces (RQ5): target position the agent selects at each
S_grid = np.linspace(0.7, 1.3, 61)         
t_grid = np.linspace(0.0, 0.95, 40)         

delta_surf = np.array([[BSDelta(m * K, K, T * (1 - tf), r, sigma)
                        for m in S_grid] for tf in t_grid])


def save_heatmap(surf, fname):
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    im = ax.pcolormesh(S_grid, t_grid, surf, vmin=0, vmax=1, shading="auto")
    ax.set_xlabel("$S/K$")
    ax.set_ylabel("$t/N$")
    fig.colorbar(im, ax=ax, label="Position")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, fname + ".png"), dpi=300)
    plt.savefig(os.path.join(OUT, fname + ".pdf"))
    plt.close(fig)


save_heatmap(delta_surf, "surface_bsdelta")

RHO_PREV_FIXED = 0.0    # query target position from a flat holding

for reg, cfg in SELECTED.items():
    model, vecnorm = models[reg]
    dummy = DummyVecEnv([lambda beta=cfg["beta"]: HedgingEnv(beta=beta)])
    vn = VecNormalize.load(vecnorm, dummy)
    vn.training = False

    policy_surf = np.zeros_like(delta_surf)
    for i, tf in enumerate(t_grid):
        for j, m in enumerate(S_grid):
            rho_prev = RHO_PREV_FIXED
            obs = np.array([[m, tf, rho_prev]], dtype=np.float32)
            obs_n = vn.normalize_obs(obs)
            a, _ = model.predict(obs_n, deterministic=True)
            a_clipped = np.clip(a[0, 0], -rho_prev, 1.0 - rho_prev)
            policy_surf[i, j] = np.clip(rho_prev + a_clipped, 0.0, 1.0)
    save_heatmap(policy_surf, f"surface_ppo_{reg}")
    print(f"Saved surface_ppo_{reg}")