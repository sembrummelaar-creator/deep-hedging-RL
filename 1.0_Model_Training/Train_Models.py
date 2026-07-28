# Installing Packages
import csv
import argparse
import os
import statistics
import time
import itertools
import numpy as np
import warnings
import gymnasium as gym
from gymnasium import spaces
from scipy.stats import norm
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize

# Environment Parameters
r = 0.0455
mu = r
sigma = 0.30
T = 0.25
N = 63
dt = T / N
S0 = 100
K = 100
Beta_NoCost = 0.0
Beta_Cost = 0.005

# Agent Parameters
RANDOM_SEED = 42
N_ENVS = 8
GAMMA = 1.0
GAE_LAMBDA = 0.95
BATCH_SIZE = 252
N_STEPS_ROLLOUT = 4 * N
N_EPOCHS = 5
ENT_COEF = 0.005
TOTAL_TIMESTEPS = 1_008_000
EVAL_FREQ = 10_000
N_EVAL_EPISODES = 100

warnings.filterwarnings(
    "ignore",
    message="Training and eval env are not of the same type",
    category=UserWarning,
)
assert (N_STEPS_ROLLOUT * N_ENVS) % BATCH_SIZE == 0, "batch_size must divide n_envs * n_steps"

# Parameter Arrays for Search
SAVE_PATH = "1.0_Model_Training/Trained_Models"
LEARNING_RATE = [1e-4, 3e-4, 1e-3]
CLIP_RANGE = [0.1, 0.2, 0.3]
BETAS = {"nocost": Beta_NoCost, "cost": Beta_Cost}


# Auxiliary Functions
def BSDelta(S, K, tau, r, sigma):
    if tau <= 0:
        return 1.0 if S > K else 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))
    return norm.cdf(d1)

def BSPrice(S, K, tau, r, sigma):
    if tau <= 0:
        return max(S - K, 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))
    d2 = d1 - sigma * np.sqrt(tau)
    return S * norm.cdf(d1) - K * np.exp(-r * tau) * norm.cdf(d2)


# Model Training Environment
class HedgingEnv(gym.Env):
    """
    State:  (S(t)/K, t/N, rho(t-1))                        
    Action: a_t in [-1, 1]
            constrained: -rho(t-1) <= a_t <= 1-rho(t-1)    
    Transition: rho(t) = rho(t-1) + a_t                    
                B(t) = B(t-1)e^{r dt} - a_t S(t) - c(t)    
    Reward: -c(t)/K for t<N,
            -lambda_T * |Pi(T)|/K at maturity with
            Pi(T) = B(T) + rho(N)S(T) - c(T) - payoff      -
    Bank account initialized with the BS premium V(0),
    """

    def __init__(self, K=K, S0=S0, sigma=sigma, r=r, mu=mu, T=T, N=N, beta=Beta_NoCost, lambda_T=1.0, sigma_price=None, seed=None):
        super().__init__()
        self.K = K
        self.S0 = S0
        self.sigma = sigma
        self.r = r
        self.mu = mu
        self.T = T
        self.N = N
        self.dt = T / N
        self.beta = beta
        self.lambda_T = lambda_T
        self._rng = np.random.default_rng(seed)
        self.sigma = sigma                              
        self.sigma_price = sigma_price if sigma_price is not None else sigma

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([np.inf, 1.0, 1.0], dtype=np.float32),
        )

        self.path = None
        self.t_idx = 0
        self.rho = 0.0
        self.B = 0.0

    def _simulate_path(self):
        dW = self._rng.normal(0, np.sqrt(self.dt), size=self.N)
        W = np.concatenate([[0.0], np.cumsum(dW)])
        t_grid = np.linspace(0, self.T, self.N + 1)
        return self.S0 * np.exp((self.mu - 0.5 * self.sigma**2) * t_grid + self.sigma * W)

    def _get_state(self):
        S_t = self.path[self.t_idx]
        return np.array([S_t / self.K, self.t_idx / self.N, self.rho], dtype=np.float32)

    def reset(self, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.path = self._simulate_path()
        self.t_idx = 0
        self.rho = 0.0
        self.B = BSPrice(self.S0, self.K, self.T, self.r, self.sigma_price)   # premium-funded
        return self._get_state(), {}

    def step(self, action):
        a_t = float(np.clip(action[0], -self.rho, 1.0 - self.rho))
        S_t = self.path[self.t_idx]

        c_t = self.beta * abs(a_t) * S_t
        self.B = self.B * np.exp(self.r * self.dt) - a_t * S_t - c_t
        self.rho = self.rho + a_t
        self.t_idx += 1
        terminated = self.t_idx == self.N

        if not terminated:
            reward = -c_t / self.K
            info = {"c_t": c_t}
        else:
            S_T = self.path[self.t_idx]
            c_T = self.beta * self.rho * S_T
            payoff = max(S_T - self.K, 0.0)
            Pi_T = self.B + self.rho * S_T - c_T - payoff
            reward = -self.lambda_T * abs(Pi_T) / self.K
            info = {"c_t": c_t, "c_T": c_T, "Pi_T": Pi_T, "B_T": self.B, "S_T_price": S_T}
            
        return self._get_state(), reward, terminated, False, info

# create environment
def make_single_env(beta, seed, lambda_T=1.0):
    def _init():
        return HedgingEnv(K=K, S0=S0, sigma=sigma, r=r, mu=mu, T=T, N=N,
                          beta=beta, lambda_T=lambda_T, seed=seed)
    return _init

# Paralel environments
def make_vec_train_env(beta, base_seed, lambda_T=1.0, n_envs=N_ENVS):
    return SubprocVecEnv([make_single_env(beta, base_seed + i, lambda_T) for i in range(n_envs)])

# formatting ETA
def format_duration(seconds):
    hrs, rem = divmod(int(seconds), 3600)
    mins, secs = divmod(rem, 60)
    if hrs > 0:
        return f"{hrs}h {mins}m {secs}s"
    elif mins > 0:
        return f"{mins}m {secs}s"
    return f"{secs}s"


# Training (single run, given a fixed learning rate / clip range)
def train_model(beta, name, learning_rate, clip_range, lambda_T=1.0,
                total_timesteps=TOTAL_TIMESTEPS, save_path=SAVE_PATH):
    train_env = make_vec_train_env(beta, base_seed=RANDOM_SEED, lambda_T=lambda_T, n_envs=N_ENVS)
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=False, gamma=GAMMA)

    eval_env = DummyVecEnv([
        lambda: HedgingEnv(K=K, S0=S0, sigma=sigma, r=r, mu=mu, T=T, N=N,
                           beta=beta, lambda_T=lambda_T, seed=RANDOM_SEED + 1000)
    ])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, training=False)
    eval_env.obs_rms = train_env.obs_rms

    os.makedirs(save_path, exist_ok=True)
    best_model_path = os.path.join(save_path, name)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=best_model_path,
        log_path=os.path.join(save_path, "logs", name),
        eval_freq=max(EVAL_FREQ // N_ENVS, 1),
        n_eval_episodes=N_EVAL_EPISODES,
        deterministic=True,
        verbose=0,
    )

    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=learning_rate,
        clip_range=clip_range,
        gamma=GAMMA,
        gae_lambda=GAE_LAMBDA,
        batch_size=BATCH_SIZE,
        n_steps=N_STEPS_ROLLOUT,
        n_epochs=N_EPOCHS,
        ent_coef=ENT_COEF,
        seed=RANDOM_SEED,
        policy_kwargs=dict(net_arch=[64, 64], log_std_init=-1.5),
        verbose=0,
    )

    start = time.time()
    model.learn(total_timesteps=total_timesteps, callback=eval_callback)
    elapsed = time.time() - start

    mean_reward = eval_callback.best_mean_reward
    final_path = os.path.join(save_path, f"{name}_final")
    model.save(final_path)

    vecnorm_path = os.path.join(save_path, f"{name}_vecnormalize.pkl")
    train_env.save(vecnorm_path)

    train_env.close()
    eval_env.close()

    return model, mean_reward, elapsed


def train_all_combinations(total_timesteps=TOTAL_TIMESTEPS, save_path=SAVE_PATH):
    combinations = list(itertools.product(BETAS.items(), LEARNING_RATE, CLIP_RANGE))
    n_total = len(combinations)

    os.makedirs(save_path, exist_ok=True)
    results_path = os.path.join(save_path, "grid_results.csv")

    # Create CSV with header only if it doesn't exist yet (so re-launches append)
    if not os.path.exists(results_path):
        with open(results_path, "w", newline="") as f:
            csv.writer(f).writerow(
                ["run_name", "beta_label", "beta", "lr", "clip_range",
                 "best_eval_reward", "elapsed_s"]
            )

    run_times = []

    for i, ((cost_label, beta), lr, cr) in enumerate(combinations, start=1):
        run_name = f"ppo_{cost_label}_lr{lr}_cr{cr}"

        # Skip runs that already finished (crash / re-launch resilience)
        if os.path.exists(os.path.join(save_path, f"{run_name}_final.zip")):
            print(f"[{i}/{n_total}] {run_name} | already trained, skipping")
            continue

        if run_times:
            typical_time = statistics.median(run_times)
            remaining = typical_time * (n_total - i + 1)
            eta_str = format_duration(remaining)
        else:
            eta_str = "calculating..."

        print(f"[{i}/{n_total}] {run_name} | ETA remaining: {eta_str}")

        _, mean_reward, elapsed = train_model(
            beta=beta, name=run_name, learning_rate=lr, clip_range=cr,
            total_timesteps=total_timesteps, save_path=save_path,
        )
        run_times.append(elapsed)

        with open(results_path, "a", newline="") as f:
            csv.writer(f).writerow(
                [run_name, cost_label, beta, lr, cr,
                 round(mean_reward, 6), round(elapsed, 1)]
            )

        print(f"[{i}/{n_total}] {run_name} | reward: {mean_reward:.6f}")

    print(f"Done. Total time: {format_duration(sum(run_times))}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=TOTAL_TIMESTEPS)
    args = parser.parse_args()

    train_all_combinations(total_timesteps=args.timesteps)