import os
import gym
from env.custom_hopper import *
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

LOG_DIR = "./models_udr"
os.makedirs(LOG_DIR, exist_ok=True)

def train_and_evaluate_udr():
    env_id = "CustomHopper-source-v0"
    train_env = gym.make(env_id)
    test_env_source = gym.make(env_id)
    test_env_target = gym.make("CustomHopper-target-v0")

    print("Dynamics (source):", train_env.get_parameters())
    print("Dynamics (target):", test_env_target.get_parameters())

    # Create PPO model
    model = PPO(
        "MlpPolicy",
        train_env,
        verbose=1,
        learning_rate=3e-4,
        gamma=0.99,
        n_steps=2048,
        ent_coef=0.01,
        batch_size=64,
        tensorboard_log="./ppo_udr_tensorboard/"
    )

    # Train model with UDR active
    model.learn(total_timesteps=1_000_000)
    model_path = os.path.join(LOG_DIR, "ppo_udr")
    model.save(model_path)

    # Evaluate policy on source
    m_src, s_src = evaluate_policy(model, test_env_source, n_eval_episodes=50, deterministic=True)
    print(f"\n[UDR Evaluation] source → source: {m_src:.2f} ± {s_src:.2f}")

    # Evaluate policy on target
    m_tgt, s_tgt = evaluate_policy(model, test_env_target, n_eval_episodes=50, deterministic=True)
    print(f"[UDR Evaluation] source → target: {m_tgt:.2f} ± {s_tgt:.2f}")

if __name__ == "__main__":
    train_and_evaluate_udr()
