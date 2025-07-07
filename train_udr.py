import os
import gym
import pandas as pd
import numpy as np
import random
import torch
from env.custom_hopper import *
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

# Fix seed for reproducibility
seed = 2025
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

LOG_DIR = "./models"
os.makedirs(LOG_DIR, exist_ok=True)

def load_best_hyperparameters(csv_path='optimization_results.csv'):
    try:
        df = pd.read_csv(csv_path)
        print(f"Trovati {len(df)} trial nell'ottimizzazione")
    except Exception as e:
        raise ValueError(f"Errore nel caricamento del file CSV: {e}")
    completed_trials = df[df['state'] == 'COMPLETE'].copy()
    best_trial_idx = completed_trials['value'].idxmax()
    best_trial = completed_trials.loc[best_trial_idx]
    best_params = {}
    param_columns = [col for col in df.columns if col.startswith('params_')]
    for col in param_columns:
        param_name = col.replace('params_', '')
        param_value = best_trial[col]
        if isinstance(param_value, float):
            param_value = round(param_value, 5)
        best_params[param_name] = param_value
    return best_params

def PPO_agent(model_name, env_train, optimization_results_path='optimization_results.csv'):
    env_id_train = env_train.spec.id
    print(f"\nTraining on {env_id_train}")
    best_params = load_best_hyperparameters(optimization_results_path)
    base_lr = best_params.get('learning_rate', 3e-4)
    total_timesteps = 1_000_000

    # Funzione per decrescita a step del learning rate
    def lr_schedule(progress_remaining):
        current_step = int((1 - progress_remaining) * total_timesteps)
        factor = current_step // 352356  # dimezza ogni 352356 step
        return base_lr * (0.5 ** factor)
    
    # Crea il modello con gli iperparametri ottimizzati
    model = PPO(
        policy="MlpPolicy",
        env=env_train,
        learning_rate=lr_schedule,
        n_steps=int(best_params.get('n_steps', 2048)),
        batch_size=int(best_params.get('batch_size', 64)),
        n_epochs=int(best_params.get('n_epochs', 10)),
        gamma=best_params.get('gamma', 0.99),
        gae_lambda=best_params.get('gae_lambda', 0.95),
        clip_range=best_params.get('clip_range', 0.2),
        ent_coef=best_params.get('ent_coef', 0.1),
        normalize_advantage=True,
        tensorboard_log=f"./{model_name}_tensorboard/",
        verbose=1
    )

    model.learn(total_timesteps=1_000_000)
    model.save(os.path.join(LOG_DIR, model_name))

def main():
    source_env = gym.make('CustomHopper-source-v0')
    target_env = gym.make('CustomHopper-target-v0')
    source_udr_env = gym.make('CustomHopper-source-udr-v0')

    print('State space:', source_env.observation_space)
    print('Action space:', source_env.action_space)
    print('Dynamics parameters (source):', source_env.get_parameters())
    print('Dynamics parameters (target):', target_env.get_parameters())
    print('UDR enabled (source-udr):', source_udr_env.enable_udr)

    # Choose which environment to train on
    # Set train_mode to select training configuration:
    # 'source': train on source domain
    # 'target': train on target domain  
    # 'source_udr': train on source with UDR enabled
    train_mode = 'source_udr'
    
    if train_mode == 'source':
        print("\n=== Training Agent on SOURCE domain ===")
        PPO_agent("ppo_source", source_env, optimization_results_path='optimization_results.csv')
        print("\n=== Testing SOURCE-trained agent on SOURCE domain ===")
        model = PPO.load(os.path.join(LOG_DIR, "ppo_source"))
        mean_reward, std_reward = evaluate_policy(model, source_env, n_eval_episodes=50, deterministic=True)
        print(f"source → source: {mean_reward:.2f} ± {std_reward:.2f}")
        
    elif train_mode == 'target':
        print("\n=== Training Agent on TARGET domain ===")
        PPO_agent("ppo_target", target_env, optimization_results_path='optimization_results.csv')
        print("\n=== Testing TARGET-trained agent on TARGET domain ===")
        model = PPO.load(os.path.join(LOG_DIR, "ppo_target"))
        mean_reward, std_reward = evaluate_policy(model, target_env, n_eval_episodes=50, deterministic=True)
        print(f"target → target: {mean_reward:.2f} ± {std_reward:.2f}")
        
    elif train_mode == 'source_udr':
        print("\n=== Training Agent on SOURCE domain with UDR ===")
        PPO_agent("ppo_source_udr", source_udr_env, optimization_results_path='optimization_results.csv')
        print("\n=== Testing SOURCE-UDR-trained agent ===")
        model = PPO.load(os.path.join(LOG_DIR, "ppo_source_udr"))
        
        # Test on source domain
        mean_reward_src, std_reward_src = evaluate_policy(model, source_env, n_eval_episodes=50, deterministic=True)
        print(f"source_udr → source: {mean_reward_src:.2f} ± {std_reward_src:.2f}")
        
        # Test on target domain
        mean_reward_tgt, std_reward_tgt = evaluate_policy(model, target_env, n_eval_episodes=50, deterministic=True)
        print(f"source_udr → target: {mean_reward_tgt:.2f} ± {std_reward_tgt:.2f}")

if __name__ == '__main__':
    main()
