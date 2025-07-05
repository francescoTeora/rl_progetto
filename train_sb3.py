"""Sample script for training a control policy on the Hopper environment
   using stable-baselines3 (https://stable-baselines3.readthedocs.io/en/master/)

    Read the stable-baselines3 documentation and implement a training
    pipeline with an RL algorithm of your choice between PPO and SAC.
"""
import os
import gym
import pandas as pd
from env.custom_hopper import *
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

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
        # Rimuovi il prefisso 'params_' per ottenere il nome pulito del parametro
        param_name = col.replace('params_', '')
        param_value = best_trial[col]
        best_params[param_name] = param_value
    return best_params
    
    
def PPO_agent( model_name ,env_train, 
              env_test, optimization_results_path='optimization_results.csv'):
   
   #optimization_results_path to use the best hyperparameters found

    env_id_train = env_train.spec.id
    env_id_test = env_test.spec.id
    print(f"\nTraining on {env_id_train}, testing on {env_id_test}")
    best_params = load_best_hyperparameters(optimization_results_path)
                
    # Crea il modello con gli iperparametri ottimizzati
    # Create model
    model = PPO(
            policy="MlpPolicy",  # Policy network architecture
            env=env_train,       # Training environment
            learning_rate=best_params.get('learning_rate', 3e-4),
            n_steps=int(best_params.get('n_steps', 2048)),
            batch_size=int(best_params.get('batch_size', 64)),
            n_epochs=int(best_params.get('n_epochs', 10)),
            gamma=best_params.get('gamma', 0.99),
            gae_lambda=best_params.get('gae_lambda', 0.95),
            clip_range=best_params.get('clip_range', 0.2),
            ent_coef=best_params.get('ent_coef', 0.0),
            normalize_advantage=True,
            tensorboard_log=f"./{model_name}_tensorboard/",
            verbose=1
        )
    # Train
    model.learn(total_timesteps=100_000)
    model.save(os.path.join(LOG_DIR, model_name))

    # Evaluate
    mean_reward, std_reward = evaluate_policy(
        model, env_test, n_eval_episodes=50, deterministic=True, render= True)
    
    print(f"Evaluation on {env_id_test}: Mean reward = {mean_reward:.2f} +/- {std_reward:.2f}")
    return mean_reward, std_reward


def main():
    source_env = gym.make('CustomHopper-source-v0')
    target_env = gym.make('CustomHopper-target-v0')

    print('State space:', source_env.observation_space)
    print('Action space:', source_env.action_space)
    print('Dynamics parameters:', source_env.get_parameters())

    # TRAIN Agent 1: source domain
    print("\n=== Training Agent on SOURCE domain ===")
    PPO_agent("ppo_source",source_env, source_env, optimization_results_path='optimization_results.csv')
    # TEST Agent 1: source -> source
    print("\n=== Testing SOURCE-trained agent on SOURCE domain ===")
    source_model = PPO.load(os.path.join(LOG_DIR, "ppo_source"))
    m1, s1 = evaluate_policy(source_model, target_env, n_eval_episodes=50, deterministic=True)
    
    # TEST Agent 1: source -> target (lower bound)
    print("\n=== Testing SOURCE-trained agent on TARGET domain ===")
    m2, s2 = evaluate_policy(source_model, target_env, n_eval_episodes=50, deterministic=True)
    
    # TRAIN Agent 2: target domain  
    print("\n=== Training Agent on TARGET domain ===")
    PPO_agent("ppo_target", target_env, target_env, optimization_results_path='optimization_results.csv')
     # TEST Agent 2: target -> target (upper bound)
    print("\n=== Testing TARGET-trained agent on TARGET domain ===")
    target_model = PPO.load(os.path.join(LOG_DIR, "ppo_target"))
    m3, s3 = evaluate_policy(target_model, target_env, n_eval_episodes=50, deterministic=True)
    print("\n" + "="*50)
    print("FINAL RESULTS SUMMARY")
    print("="*50)
    print(f"source → source: {m1:.2f} ± {s1:.2f}")
    print(f"source → target: {m2:.2f} ± {s2:.2f} (LOWER BOUND)")
    print(f"target → target: {m3:.2f} ± {s3:.2f} (UPPER BOUND)")
    print("="*50)

if __name__ == '__main__':
    main()
