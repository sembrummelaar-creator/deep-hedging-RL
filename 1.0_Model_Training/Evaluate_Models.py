import os
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from Train_Models import HedgingEnv, BSDelta, BSPrice, K, S0, sigma, r, mu, T, N, Beta_NoCost

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_NAME = "ppo_cost_lambda10_lr0.0003_cr0.2_final"
N_ROLLOUTS = 20

# model_path = os.path.join(SCRIPT_DIR, "Trained_Models", "ppo_nocost_lr0.0003_cr0.2", "best_model")
model_path = os.path.join(SCRIPT_DIR, "Trained_Models", MODEL_NAME)
vecnorm_path = os.path.join(SCRIPT_DIR, "Trained_Models", f"{MODEL_NAME.replace('_final', '')}_vecnormalize.pkl")

model = PPO.load(model_path)


def make_normalized_env(seed):
    raw_env = DummyVecEnv([
        lambda: HedgingEnv(K=K, S0=S0, sigma=sigma, r=r, mu=mu, T=T, N=N, beta=Beta_NoCost, seed=seed)
    ])
    env = VecNormalize.load(vecnorm_path, raw_env)
    env.training = False
    env.norm_reward = False
    return env


def rollout(seed):
    env = make_normalized_env(seed)
    obs = env.reset()
    done = False

    rho_path, delta_path, tau_path = [], [], []

    while not done:
        raw_obs = env.get_original_obs()[0]
        S_ratio, t_frac, rho_prev = raw_obs
        S_t = S_ratio * K
        tau = T - t_frac * T
        delta_t = BSDelta(S_t, K, tau, r, sigma)

        rho_path.append(rho_prev)
        delta_path.append(delta_t)
        tau_path.append(t_frac)

        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done_arr, info = env.step(action)
        done = done_arr[0]

    rho_path, delta_path = np.array(rho_path), np.array(delta_path)
    return {
        "rho_path": rho_path,
        "delta_path": delta_path,
        "tau_path": np.array(tau_path),
        "Pi_T": info[0]["Pi_T"],
        "B_T": info[0]["B_T"],
        "epsilon_cum": np.sum(np.abs(rho_path - delta_path)),
    }



# Run N_ROLLOUTS rollouts and aggregate
results = [rollout(seed=1000 + i) for i in range(N_ROLLOUTS)]

epsilon_cums = np.array([res["epsilon_cum"] for res in results])
Pi_Ts = np.array([res["Pi_T"] for res in results])

V0 = BSPrice(S0, K, T, r, sigma)

print(f"Over {N_ROLLOUTS} rollouts:")
print(f"  BS premium V(0):        {V0:.4f}")
print(f"  Mean epsilon_cum:       {epsilon_cums.mean():.4f}  (std: {epsilon_cums.std():.4f})")
print(f"\nP&L statistics, Pi(T):")
print(f"  Mean Pi(T):             {Pi_Ts.mean():.4f}")
print(f"  Std Pi(T):              {Pi_Ts.std():.4f}")
print(f"  Mean |Pi(T)|:           {np.mean(np.abs(Pi_Ts)):.4f}")
print(f"  RMSE Pi(T):             {np.sqrt(np.mean(Pi_Ts**2)):.4f}")
print(f"  Min / Max Pi(T):        {Pi_Ts.min():.4f} / {Pi_Ts.max():.4f}")
print(f"\nRelative to strike (per unit K):")
print(f"  Mean |Pi(T)|/K:         {np.mean(np.abs(Pi_Ts))/K:.6f}")
print(f"  RMSE Pi(T)/K:           {np.sqrt(np.mean(Pi_Ts**2))/K:.6f}")


# Bucket by time-to-maturity (t/T)
n_buckets = 10
bucket_edges = np.linspace(0, 1, n_buckets + 1)
bucket_errors = [[] for _ in range(n_buckets)]

for res in results:
    diffs = np.abs(res["rho_path"] - res["delta_path"])
    for tau_frac, diff in zip(res["tau_path"], diffs):
        bucket_idx = min(int(tau_frac * n_buckets), n_buckets - 1)
        bucket_errors[bucket_idx].append(diff)

print("\nMean |rho - Delta| by time-to-maturity bucket:")
for i, errs in enumerate(bucket_errors):
    if errs:
        lo, hi = bucket_edges[i], bucket_edges[i + 1]
        print(f"  t/T in [{lo:.1f}, {hi:.1f}): mean diff = {np.mean(errs):.4f}  (n={len(errs)})")

# Plots: rho vs Delta | error by time | Pi(T) distribution
fig, axes = plt.subplots(1, 3, figsize=(16, 4))

for res in results[:5]:
    axes[0].plot(res["tau_path"], res["rho_path"], alpha=0.6, color="tab:blue")
    axes[0].plot(res["tau_path"], res["delta_path"], alpha=0.6, color="tab:orange", linestyle="--")
axes[0].set_xlabel("t/T")
axes[0].set_ylabel("Position")
axes[0].set_title("rho(t) [blue] vs Delta(t) [orange, dashed]")

bucket_means = [np.mean(e) if e else np.nan for e in bucket_errors]
bucket_centers = (bucket_edges[:-1] + bucket_edges[1:]) / 2
axes[1].bar(bucket_centers, bucket_means, width=0.08)
axes[1].set_xlabel("t/T")
axes[1].set_ylabel("Mean |rho - Delta|")
axes[1].set_title("Hedging error by time-to-maturity")

axes[2].hist(Pi_Ts, bins=15, color="tab:blue", edgecolor="white")
axes[2].axvline(0.0, color="black", linestyle="--", linewidth=1)
axes[2].axvline(Pi_Ts.mean(), color="tab:red", linestyle="-", linewidth=1.5,
                label=f"mean = {Pi_Ts.mean():.2f}")
axes[2].set_xlabel("Pi(T)")
axes[2].set_ylabel("Count")
axes[2].set_title("Terminal P&L distribution")
axes[2].legend()

plt.tight_layout()
plt.savefig(os.path.join(SCRIPT_DIR, "diagnostic_plot.png"), dpi=150)
print(f"\nPlot saved to diagnostic_plot.png")