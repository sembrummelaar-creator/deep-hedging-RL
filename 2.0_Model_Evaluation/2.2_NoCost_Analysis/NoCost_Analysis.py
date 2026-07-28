import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Common import SELECTED, load_model, evaluate_paired, pnl_stats, K

REGIME = "nocost"         
OUT = os.path.dirname(os.path.abspath(__file__))
TAG = REGIME

cfg = SELECTED[REGIME]
model, vecnorm = load_model(cfg["run_name"])
agent, bench = evaluate_paired(model, vecnorm, cfg["beta"])


# P&L table

sa, sb = pnl_stats(agent), pnl_stats(bench)
diff = sa["Pi"] - sb["Pi"]
t_stat, p_val = stats.ttest_rel(sa["Pi"], sb["Pi"])
w_stat, w_p = stats.wilcoxon(sa["Pi"], sb["Pi"])

rows = []
for name, s in [("PPO", sa), ("BS Delta", sb)]:
    rows.append({"strategy": name, "mean": s["mean"], "ci95": s["ci95"],
                 "std": s["std"], "mean_abs": s["mean_abs"],
                 "mean_cost": s["mean_cost"], "min": s["min"], "max": s["max"]})
rows.append({"strategy": "Paired diff", "mean": diff.mean(),
             "ci95": 1.96 * diff.std(ddof=1) / np.sqrt(len(diff)),
             "std": diff.std(ddof=1), "mean_abs": np.nan, "mean_cost": np.nan,
             "min": np.nan, "max": np.nan})

df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUT, f"pnl_{TAG}.csv"), index=False)
print(df.to_string(index=False))
print(f"\nPaired t-test:  t = {t_stat:.4f}, p = {p_val:.4g}")
print(f"Wilcoxon:       W = {w_stat:.1f}, p = {w_p:.4g}")


# Sanity check on terminal prices (should be roughly [75, 135], mean ~101)
S_T = np.array([res["S_T"] for res in agent])
print(f"\nS_T: min={S_T.min():.2f}, max={S_T.max():.2f}, mean={S_T.mean():.2f}")
print(f"Fraction of paths finishing ITM (S_T > K): {(S_T > K).mean():.2%}")

# Representative paths by terminal moneyness: OTM / ATM / ITM
S_T = np.array([res["S_T"] for res in agent])
moneyness = S_T / K
print(f"\nS_T: min={S_T.min():.2f}, max={S_T.max():.2f}, mean={S_T.mean():.2f}")
print(f"Fraction finishing ITM (S_T > K): {(S_T > K).mean():.2%}")

order = np.argsort(moneyness)
idx_otm = int(order[int(0.10 * len(order))])   # 10th percentile: clearly OTM
idx_atm = int(order[int(0.50 * len(order))])   # median: near the money
idx_itm = int(order[int(0.90 * len(order))])   # 90th percentile: clearly ITM

for idx, tag in [(idx_otm, "otm"), (idx_atm, "atm"), (idx_itm, "itm")]:
    res = agent[idx]
    mny = res["S_T"] / K
    print(f"{tag.upper()} path: seed index {idx}, terminal S_T/K = {mny:.3f}")
    t_grid = np.arange(len(res["rho"])) / len(res["rho"])
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    ax.plot(t_grid, res["rho"], color="tab:blue", label=r"$\rho(t)$")
    ax.plot(t_grid, res["delta"], color="tab:orange", ls="--", label=r"$\Delta(t)$")
    ax2 = ax.twinx()
    ax2.plot(t_grid, res["S"], color="tab:gray", alpha=0.5, lw=0.9)
    ax2.set_ylabel("$S(t)$")
    ax.set_xlabel("$t/N$")
    ax.set_ylabel("Position")
    ax.legend(fontsize=7, loc="best")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, f"path_{tag}_{TAG}.png"), dpi=300)
    plt.savefig(os.path.join(OUT, f"path_{tag}_{TAG}.pdf"))
    plt.close(fig)

for idx, tag in [(idx_otm, "otm"), (idx_atm, "atm"), (idx_itm, "itm")]:
    res = agent[idx]
    print(f"[CHECK] {tag}: printed S_T/K={res['S_T']/K:.3f}, "
          f"S array last={res['S'][-1]:.2f}, S array first={res['S'][0]:.2f}, "
          f"len(S)={len(res['S'])}")


# Terminal P&L histogram, PPO vs BS Delta
fig, ax = plt.subplots(figsize=(3.5, 2.8))
bins = np.histogram_bin_edges(np.concatenate([sa["Pi"], sb["Pi"]]), bins=40)
ax.hist(sa["Pi"], bins=bins, alpha=0.6, label="PPO")
ax.hist(sb["Pi"], bins=bins, alpha=0.6, label="BS Delta")
ax.axvline(0, color="black", ls="--", lw=1)
ax.set_xlabel(r"$\Pi(T)$")
ax.set_ylabel("Count")
ax.legend(fontsize=7)
plt.tight_layout()
plt.savefig(os.path.join(OUT, f"hist_{TAG}.png"), dpi=300)
plt.savefig(os.path.join(OUT, f"hist_{TAG}.pdf"))
plt.close(fig)

print(f"Saved pnl_{TAG}.csv, path_otm_{TAG}, path_atm_{TAG}, path_itm_{TAG}, hist_{TAG}")