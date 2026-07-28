import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# locate project and import training env
EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(EVAL_DIR)
TRAIN_DIR = os.path.join(PROJECT_DIR, "1.0_Model_Training")
MODELS_DIR = os.path.join(TRAIN_DIR, "Trained_Models")
sys.path.insert(0, TRAIN_DIR)

from Train_Models import (HedgingEnv, BSDelta, BSPrice, r, mu, T, N)
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# configuration
RUN_NAME = "ppo_nocost_lr0.0003_cr0.2"       
SIGMA_PRICE = 0.30                            
DATA_PATH = os.path.join(EVAL_DIR, "real_path.csv")
OUT = EVAL_DIR


#  env that replays a fixed real path instead of simulating GBM
class RealPathHedgingEnv(HedgingEnv):
    def __init__(self, real_path, **kwargs):
        self._real_path = np.asarray(real_path, dtype=float)
        super().__init__(**kwargs)

    def _simulate_path(self):
        return self._real_path


def load_model(run_name):
    model = PPO.load(os.path.join(MODELS_DIR, run_name, "best_model"))
    vecnorm_path = os.path.join(MODELS_DIR, f"{run_name}_vecnormalize.pkl")
    return model, vecnorm_path


def rollout_agent_real(model, vecnorm_path, real_path, K_real, sigma_price):
    env = RealPathHedgingEnv(real_path, K=K_real, S0=real_path[0],
                             sigma=sigma_price, r=r, mu=mu, T=T, N=N,
                             beta=0.0, lambda_T=1.0, sigma_price=sigma_price,
                             seed=0)
    venv = DummyVecEnv([lambda: env])
    venv = VecNormalize.load(vecnorm_path, venv)
    venv.training = False
    venv.norm_reward = False
    obs = venv.reset()

    rho_p, delta_p, S_p = [], [], []
    step_i = 0
    done = False
    while not done:
        _, _, rho_prev = venv.get_original_obs()[0]
        S_t = real_path[step_i]
        tau = T - (step_i / N) * T
        rho_p.append(rho_prev)
        delta_p.append(BSDelta(S_t, K_real, tau, r, sigma_price))
        S_p.append(S_t)
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done_arr, info = venv.step(action)
        done = done_arr[0]
        step_i += 1
    return {"rho": np.array(rho_p), "delta": np.array(delta_p),
            "S": np.array(S_p), "Pi_T": info[0]["Pi_T"]}


def rollout_bs_real(real_path, K_real, sigma_price):
    env = RealPathHedgingEnv(real_path, K=K_real, S0=real_path[0],
                             sigma=sigma_price, r=r, mu=mu, T=T, N=N,
                             beta=0.0, lambda_T=1.0, sigma_price=sigma_price,
                             seed=0)
    obs, _ = env.reset()
    rho_p, delta_p, S_p = [], [], []
    step_i = 0
    done = False
    while not done:
        _, _, rho_prev = obs
        S_t = real_path[step_i]
        tau = T - (step_i / N) * T
        delta_t = BSDelta(S_t, K_real, tau, r, sigma_price)
        rho_p.append(rho_prev)
        delta_p.append(delta_t)
        S_p.append(S_t)
        obs, _, done, _, info = env.step(np.array([delta_t - rho_prev]))
        step_i += 1
    return {"rho": np.array(rho_p), "delta": np.array(delta_p),
            "S": np.array(S_p), "Pi_T": info["Pi_T"]}


def main():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"{DATA_PATH} not found. Run fetch_data.py on a machine with "
            "internet first to create it.")

    real_path = np.loadtxt(DATA_PATH, delimiter=",")
    assert len(real_path) == N + 1, \
        f"real_path has {len(real_path)} points, need {N+1}"

    K_real = real_path[0]          # ATM at inception, matching training
    model, vecnorm = load_model(RUN_NAME)

    agent = rollout_agent_real(model, vecnorm, real_path, K_real, SIGMA_PRICE)
    bs = rollout_bs_real(real_path, K_real, SIGMA_PRICE)

    S_T = real_path[-1]
    print(f"Real path: S0={real_path[0]:.2f}, S_T={S_T:.2f}, "
          f"S_T/K={S_T/K_real:.3f}")
    print(f"BS premium at inception (sigma={SIGMA_PRICE}): "
          f"{BSPrice(real_path[0], K_real, T, r, SIGMA_PRICE):.3f}")
    print(f"PPO realized Pi(T):      {agent['Pi_T']:.3f}")
    print(f"BS-delta realized Pi(T): {bs['Pi_T']:.3f}")

    # --- plot: rho vs delta vs price, both strategies ---
    t_axis = np.arange(len(agent["rho"])) / len(agent["rho"])
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(t_axis, agent["rho"], color="tab:blue", label=r"$\rho(t)$ PPO")
    ax.plot(t_axis, bs["rho"], color="tab:green", label=r"$\rho(t)$ BS delta")
    ax.plot(t_axis, agent["delta"], color="tab:orange", ls="--",
            label=r"$\Delta(t)$")
    ax2 = ax.twinx()
    ax2.plot(t_axis, agent["S"], color="tab:gray", alpha=0.5, lw=0.9)
    ax2.set_ylabel("$S(t)$")
    ax.set_xlabel("$t/N$")
    ax.set_ylabel("Position")
    ax.legend(fontsize=8, loc="best")
    ax.set_title(f"Real-data hedge (K={K_real:.0f}, $S_T/K$={S_T/K_real:.2f})")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "real_hedge.png"), dpi=200)
    plt.close(fig)
    print("Saved real_hedge.png")


if __name__ == "__main__":
    main()