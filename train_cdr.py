import os
import gym
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from env.custom_hopper import *

LOG_DIR = "./models"
os.makedirs(LOG_DIR, exist_ok=True)
model_name = 'ExtensionCDR_PPO'

class CurriculumDomainRandomizationAdaptive(gym.Wrapper):
    """
    Versione minimale di Adaptive CDR
    """
    def __init__(self, env, min_range=0.01, max_range=0.10):
        super().__init__(env)
        self.min_range = min_range
        self.max_range = max_range
        self.current_range = min_range
        
        # Tracking semplice con lista normale
        self.episode_rewards = []
        self.current_episode_reward = 0
        self.reset_count = 0
        
    def reset(self, **kwargs):
        self.reset_count += 1
        
        # Salva reward dell'episodio precedente
        if self.reset_count > 1:
            self.episode_rewards.append(self.current_episode_reward)
            
            # Mantieni solo gli ultimi 1000 episodi
            if len(self.episode_rewards) > 200:
                self.episode_rewards.pop(0)
            
            # Controlla se può aumentare difficoltà ogni 100 episodi
            if len(self.episode_rewards) >= 100 and self.reset_count % 50 == 0:
                self._check_difficulty_increase()
        
        # Reset per nuovo episodio
        self.current_episode_reward = 0
        
        # Applica randomizzazione corrente
        if hasattr(self.env, 'set_random_parameters'):
            self.env.set_random_parameters(self.current_range)
        return self.env.reset(**kwargs)
    
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.current_episode_reward += reward
        return obs, reward, done, info
    
    def _check_difficulty_increase(self):
        """
        Logica semplice: se la media degli ultimi 500 episodi è migliorata,
        aumenta la difficoltà
        """
        if self.current_range >= self.max_range:
            return  # Già al massimo
        
        # Prendi ultimi 500 episodi
        recent_rewards = self.episode_rewards[-100:]
        recent_avg = np.mean(recent_rewards)
        
        # Se abbiamo almeno 200 episodi, confronta con i 200 precedenti
        if len(self.episode_rewards) >= 200:
            previous_rewards = self.episode_rewards[-200:-100]
            previous_avg = np.mean(previous_rewards)
            
            # Se c'è miglioramento, aumenta difficoltà
            if (recent_avg > previous_avg and recent_avg>700):
                old_range = self.current_range
                self.current_range = min(self.max_range, self.current_range + 0.02)
                self.episode_rewards=[]
                
                print(f" Difficulty UP! Range: {old_range:.3f} → {self.current_range:.3f}")
                print(f"   Performance: {previous_avg:.1f} → {recent_avg:.1f}")

def train_simple_adaptive():
    """Training con CDR adattivo semplice"""
    print("Training with Adaptive CDR...")
    
    # Ambiente di training
    source_env = gym.make('CustomHopper-source-v0')
    total_timesteps =1_000_000
    # Ambiente di test
    target_env = gym.make('CustomHopper-target-v0')
    env_train = CurriculumDomainRandomizationAdaptive(source_env)


    def lr_schedule(progress_remaining):
        current_step = int((1 - progress_remaining) * total_timesteps)
        factor = current_step //352356  # dimezza ogni 352356 step
        return 1e-3 * (0.5 ** factor)
    
    def load_best_hyperparameters(csv_path='optimization_results.csv'):
        df = pd.read_csv(csv_path)
        print(f"Trovati {len(df)} trial nell'ottimizzazione")
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

    
    
    best_params = load_best_hyperparameters('optimization_results.csv')
    # base_lr = best_params.get('learning_rate', 3e-4)
    # Funzione per decrescita a step del learning rate
    #def linear_schedule(initial_value):
        #return lambda progress_remaining: progress_remaining * initial_value
    # Crea il modello con gli iperparametri ottimizzati
    # Create model
    model = PPO(
            policy="MlpPolicy",  # Policy network architecture
            env=env_train,       # Training environment
            learning_rate=linear_schedule(base_lr),
            n_steps=int(best_params.get('n_steps', 2048)),
            batch_size=int(best_params.get('batch_size', 64)),
            n_epochs=int(best_params.get('n_epochs', 10)),
            gamma=best_params.get('gamma', 0.99),
            gae_lambda=best_params.get('gae_lambda', 0.95),
            clip_range=best_params.get('clip_range', 0.2),
            ent_coef=best_params.get('ent_coef', 0.1),
            normalize_advantage=True,
            tensorboard_log=f"./curriculum_bohhhhhh_tensorboard/",
            verbose=1
        )

    model.learn(total_timesteps=1_000_000)
    model.save(os.path.join(LOG_DIR, model_name))
    
    # Valuta
    from stable_baselines3.common.evaluation import evaluate_policy
    mean_reward, std_reward = evaluate_policy(model, target_env, n_eval_episodes=10)
    
    print(f"\nResults: {mean_reward:.2f} ± {std_reward:.2f}")
    
   
    train_env.close()
    target_env.close()
    
    return mean_reward

if __name__ == "__main__":
    train_simple_adaptive()
