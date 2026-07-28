import os
import sys
import numpy as np

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(EVAL_DIR)
TRAIN_DIR = os.path.join(PROJECT_DIR, "1.0_Model_Training")
MODELS_DIR = os.path.join(TRAIN_DIR, "Trained_Models")
sys.path.insert(0, TRAIN_DIR)

from Train_Models import (HedgingEnv, BSDelta, BSPrice, K, S0, sigma, r, mu, T, N, Beta_NoCost, Beta_Cost)
from stable_baselines3 import PPO                                
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize 

# Selected configurations 
SELECTED = {
    "nocost": {"run_name": "ppo_nocost_lr0.0003_cr0.2", "beta": Beta_NoCost},
    "cost":   {"run_name": "ppo_cost_lr0.0003_cr0.2",   "beta": Beta_Cost},
}

M_PATHS = 1000
BASE_EVAL_SEED = 5000


def load_model(run_name):
    """Best checkpoint + frozen VecNormalize stats for a run."""
    model = PPO.load(os.path.join(MODELS_DIR, run_name, "best_model"))
    vecnorm_path = os.path.join(MODELS_DIR, f"{run_name}_vecnormalize.pkl")
    return model, vecnorm_path


def _wrap_env(env, vecnorm_path):
    venv = DummyVecEnv([lambda: env])
    venv = VecNormalize.load(vecnorm_path, venv)
    venv.training = False
    venv.norm_reward = False
    return venv


def rollout_agent(model, vecnorm_path, beta, seed, K_eval=K, sigma_eval=sigma,
                  sigma_price=sigma):
    env = HedgingEnv(K=K_eval, S0=S0, sigma=sigma_eval, r=r, mu=mu, T=T, N=N,
                     beta=beta, sigma_price=sigma_price, seed=seed)
    venv = _wrap_env(env, vecnorm_path)
    obs = venv.reset()
    true_path = venv.get_attr("path")[0].copy()
    rho_p, delta_p, S_p, c_sum = [], [], [], 0.0
    step_i = 0
    done = False
    while not done:
        _, _, rho_prev = venv.get_original_obs()[0]
        S_t = true_path[step_i]
        tau = T - (step_i / N) * T
        rho_p.append(rho_prev)
        delta_p.append(BSDelta(S_t, K_eval, tau, r, sigma_price))   # <-- belief vol
        S_p.append(S_t)
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done_arr, info = venv.step(action)
        c_sum += info[0].get("c_t", 0.0)
        done = done_arr[0]
        step_i += 1
    return {"rho": np.array(rho_p), "delta": np.array(delta_p),
            "S": np.array(S_p), "S_T": true_path[-1],
            "Pi_T": info[0]["Pi_T"], "c_sum": c_sum + info[0].get("c_T", 0.0)}


def rollout_bs(beta, seed, K_eval=K, sigma_eval=sigma, sigma_price=sigma):
    env = HedgingEnv(K=K_eval, S0=S0, sigma=sigma_eval, r=r, mu=mu, T=T, N=N,
                     beta=beta, sigma_price=sigma_price, seed=seed)
    obs, _ = env.reset()
    true_path = env.path.copy()
    rho_p, delta_p, S_p, c_sum = [], [], [], 0.0
    step_i = 0
    done = False
    while not done:
        _, _, rho_prev = obs
        S_t = true_path[step_i]
        tau = T - (step_i / N) * T
        delta_t = BSDelta(S_t, K_eval, tau, r, sigma_price)          
        rho_p.append(rho_prev)
        delta_p.append(delta_t)
        S_p.append(S_t)
        obs, _, done, _, info = env.step(np.array([delta_t - rho_prev]))
        c_sum += info.get("c_t", 0.0)
        step_i += 1
    return {"rho": np.array(rho_p), "delta": np.array(delta_p),
            "S": np.array(S_p), "S_T": true_path[-1],
            "Pi_T": info["Pi_T"], "c_sum": c_sum + info.get("c_T", 0.0)}


def evaluate_paired(model, vecnorm_path, beta, m=M_PATHS, K_eval=K,
                    sigma_eval=sigma, base_seed=BASE_EVAL_SEED, sigma_price=sigma):
    agent = [rollout_agent(model, vecnorm_path, beta, base_seed + i, K_eval, sigma_eval, sigma_price)
             for i in range(m)]
    bench = [rollout_bs(beta, base_seed + i, K_eval, sigma_eval, sigma_price) for i in range(m)]
    return agent, bench


def pnl_stats(results):
    Pi = np.array([res["Pi_T"] for res in results])
    c = np.array([res["c_sum"] for res in results])
    n = len(Pi)
    ci = 1.96 * Pi.std(ddof=1) / np.sqrt(n)
    return {"mean": Pi.mean(), "ci95": ci, "std": Pi.std(ddof=1),
            "mean_abs": np.abs(Pi).mean(), "mean_cost": c.mean(),
            "min": Pi.min(), "max": Pi.max(), "Pi": Pi}


def eps_cum(results):
    return np.array([np.sum(np.abs(res["rho"] - res["delta"])) for res in results])