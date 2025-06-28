"""Sample script for training a control policy on the Hopper environment
   using stable-baselines3 (https://stable-baselines3.readthedocs.io/en/master/)

    Read the stable-baselines3 documentation and implement a training
    pipeline with an RL algorithm of your choice between PPO and SAC.
"""
import os
import gym
from env.custom_hopper import *
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

LOG_DIR = "./models"
os.makedirs(LOG_DIR, exist_ok=True)

def PPO_agent(env_train, env_test, model_name, learning_rate=3e-4, gamma=0.99, verbose=1, n_step=2048, total_timesteps=1_000_000):
    env_id_train = env_train.spec.id
    env_id_test = env_test.spec.id
    print(f"\nTraining on {env_id_train}, testing on {env_id_test}")

    # Create model
    model = PPO(
        "MlpPolicy",
        env_train,
        verbose=verbose,
        learning_rate=learning_rate,
        gamma=gamma,
        n_steps=n_step,
        ent_coef=0.01,
        tensorboard_log="./ppo_tensorboard/"
    )

    # Train
    model.learn(total_timesteps=total_timesteps)
    model.save(os.path.join(LOG_DIR, model_name))

    # Evaluate
    mean_reward, std_reward = evaluate_policy(
        model, env_test, n_eval_episodes=50, deterministic=True
    )
    print(f"Evaluation on {env_id_test}: Mean reward = {mean_reward:.2f} +/- {std_reward:.2f}")
    return mean_reward, std_reward


def main():
    train_env = gym.make('CustomHopper-source-v0')
    test_env = gym.make('CustomHopper-source-v0')
    target_env = gym.make('CustomHopper-target-v0')

    print('State space:', train_env.observation_space)
    print('Action space:', train_env.action_space)
    print('Dynamics parameters:', train_env.get_parameters())

    # source -> source
    m1, s1 = PPO_agent(train_env, test_env, "ppo_source")

    # source -> target
    model = PPO.load(os.path.join(LOG_DIR, "ppo_source"))
    m2, s2 = evaluate_policy(model, target_env, n_eval_episodes=50, deterministic=True)

    # target -> target
    m3, s3 = PPO_agent(target_env, target_env, "ppo_target")

    print("\n--- Summary ---")
    print(f"source -> source: {m1:.2f} ± {s1:.2f}")
    print(f"source -> target: {m2:.2f} ± {s2:.2f} (lower bound)")
    print(f"target -> target: {m3:.2f} ± {s3:.2f} (upper bound)")

if __name__ == '__main__':
    main()
