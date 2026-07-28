import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Common import SELECTED, load_model, evaluate_paired

OUT = os.path.dirname(os.path.abspath(__file__))
SIGMAS = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

models = {reg: load_model(cfg["run_name"]) for reg, cfg in SELECTED.items()}

rows = []
for sig in SIGMAS:
    row = {"sigma": sig}
    for reg, cfg in SELECTED.items():
        model, vecnorm = models[reg]
        agent, bench = evaluate_paired(model, vecnorm, cfg["beta"], sigma_eval=sig)
        row[f"std_PPO_{reg}"] = round(np.array([r_["Pi_T"] for r_ in agent]).std(ddof=1), 4)
        row[f"std_BS_{reg}"] = round(np.array([r_["Pi_T"] for r_ in bench]).std(ddof=1), 4)
    rows.append(row)
    print(row)

df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUT, "volatility_table.csv"), index=False)

for reg in ["nocost", "cost"]:
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    ax.plot(df["sigma"], df[f"std_PPO_{reg}"], "o-", label="PPO", ms=4)
    ax.plot(df["sigma"], df[f"std_BS_{reg}"], "s--", label="BS Delta", ms=4)
    ax.axvline(0.30, color="gray", ls=":", lw=1)
    ax.set_xlabel(r"Evaluation $\sigma$")
    ax.set_ylabel(r"Std $\Pi(T)$")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, f"volatility_curve_{reg}.png"), dpi=300)
    plt.savefig(os.path.join(OUT, f"volatility_curve_{reg}.pdf"))
    plt.close(fig)

print("Saved volatility_table.csv, volatility_curve_nocost, volatility_curve_cost")