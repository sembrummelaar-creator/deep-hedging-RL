import os, sys, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Common import MODELS_DIR, SELECTED

OUT = os.path.dirname(os.path.abspath(__file__))

# ---- Table: pivot grid_results.csv ----
df = pd.read_csv(os.path.join(MODELS_DIR, "grid_results.csv"))
table = df.pivot_table(index=["lr", "clip_range"], columns="beta_label",
                       values="best_eval_reward").reset_index()
table.to_csv(os.path.join(OUT, "sensitivity_table.csv"), index=False)
print(table.to_string(index=False))

# Figures: learning curves
for regime in ["nocost", "cost"]:
    sel = SELECTED[regime]["run_name"]
    fig, ax = plt.subplots(figsize=(3.5, 2.8))          # IEEE column width ~3.5in
    for npz_path in sorted(glob.glob(os.path.join(MODELS_DIR, "logs", f"ppo_{regime}_*", "evaluations.npz"))):
        run = os.path.basename(os.path.dirname(npz_path))
        d = np.load(npz_path)
        mean_r = d["results"].mean(axis=1)
        if run == sel:
            ax.plot(d["timesteps"], mean_r, color="tab:red", lw=1.8, zorder=5, label="selected")
        else:
            ax.plot(d["timesteps"], mean_r, color="tab:blue", alpha=0.3, lw=0.9)
    ax.set_xlabel("Training timesteps")
    ax.set_ylabel("Mean evaluation reward")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, f"learning_curves_{regime}.png"), dpi=300)
    plt.close(fig)