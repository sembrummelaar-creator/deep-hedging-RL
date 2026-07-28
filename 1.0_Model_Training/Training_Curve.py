import os, glob
import numpy as np
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_ROOT = os.path.join(SCRIPT_DIR, "Trained_Models", "logs")
NEW_RUN = "ppo_cost_lambda10_lr0.0003_cr0.2"

fig, ax = plt.subplots(figsize=(8, 5))
for npz_path in sorted(glob.glob(os.path.join(LOG_ROOT, "*", "evaluations.npz"))):
    run_name = os.path.basename(os.path.dirname(npz_path))
    if "cost" not in run_name:          # skip nocost runs
        continue
    data = np.load(npz_path)
    mean_r = data["results"].mean(axis=1)
    if run_name == NEW_RUN:
        ax.plot(data["timesteps"], mean_r, color="tab:red", lw=2, zorder=5, label=f"{run_name} (λ=10)")
    else:
        ax.plot(data["timesteps"], mean_r, color="tab:blue", alpha=0.3, lw=1)

ax.set_xlabel("Training timesteps")
ax.set_ylabel("Mean evaluation reward")
ax.set_title("Cost regime: λ=10 vs λ=1")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(SCRIPT_DIR, "training_curve_cost_compare.png"), dpi=150)
print("Saved training_curve_cost_compare.png")