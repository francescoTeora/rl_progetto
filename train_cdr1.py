import os
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from env.custom_hopper import *
from env.cdr_wrapper import RandomizeEnvWrapper

LOG_DIR = "./models_cdr"
os.makedirs(LOG_DIR, exist_ok=True)

def PPO_agent(env_train, env_test, model_name, learning_rate=3e-4, gamma=0.99, verbose=1, n_step=2048, total_timesteps=100_000):
    print(f"\nTraining on {env_train}, testing on {env_test}")

    # Create and train the PPO model
    model = PPO(
        "MlpPolicy",
        train_env,
        verbose=verbose,
        learning_rate=learning_rate,
        gamma=gamma,
        n_steps=n_steps,
        ent_coef=0.01, #coefficiente che regola l'esplorazione
        tensorboard_log="./ppo_tensorboard_cdr/"
    )
#Qui alleniamo PPO
    print("Training with Curriculum Domain Randomization...")
    model.learn(total_timesteps=int(total_timesteps))
    model.save(os.path.join(LOG_DIR, model_name))

    # Evaluate on source valutiamo su 50 episodi 
    mean_reward, std_reward = evaluate_policy(model, env_test, n_eval_episodes=50, deterministic=True)
    print(f"Evaluation on test env: Mean reward = {mean_reward:.2f} +/- {std_reward:.2f}")
    return mean_reward, std_reward

#Qui alleno e confronto tutto
def main():
    train_env_source = gym.make("CustomHopper-source-v0")
    test_env_target = gym.make("CustomHopper-target-v0")
    test_env_source = gym.make("CustomHopper-source-v0")

    #addestro in source e testo in source sarebbe baseline nativa
    m1, s1 = PPO_agent(train_env_source, test_env_source, "ppo_source")
    #qui invece carichiamo la policy addestrata su source e la testiamo sull’ambiente target in modo da
    #simulare il trasferimento sim-to-real baseline senza dom rand

     model = PPO.load(os.path.join(LOG_DIR, "ppo_source"))
    m2, s2 = evaluate_policy(model, test_env_target, n_eval_episodes=50, deterministic=True)

    train_env_udr = gym.make("CustomHopper-source-v0")
    #addestriamo l'agente addestrato in un ambiente con le masse variabili quindi UDR e poi testiamo sul target
    train_env_udr.env.set_random_parameters = train_env_udr.env.set_random_parameters
    m3, s3 = PPO_agent(train_env_udr, test_env_target, "ppo_udr")

    print("\n--- Summary ---")
    print(f"source -> source: {m1:.2f} ± {s1:.2f}")
    print(f"source -> target: {m2:.2f} ± {s2:.2f}")
    print(f"UDR (source rand) -> target: {m3:.2f} ± {s3:.2f}")

if __name__ == "__main__":
    main()




import os
import gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

LOG_DIR = "./models"
os.makedirs(LOG_DIR, exist_ok=True)

def PPO_agent(env_train, env_test, model_name, total_timesteps=200_000, n_steps=2048, learning_rate=3e-4, gamma=0.99, verbose=1):
    model = PPO(
        "MlpPolicy",
        env_train,
        verbose=verbose,
        learning_rate=learning_rate,
        gamma=gamma,
        n_steps=n_steps,
        ent_coef=0.01,
        tensorboard_log="./ppo_tensorboard/"
    )

    model.learn(total_timesteps=total_timesteps)
    model.save(os.path.join(LOG_DIR, model_name))

    mean_reward, std_reward = evaluate_policy(model, env_test, n_eval_episodes=20)
    return mean_reward, std_reward


def train_and_compare():


    # ➤ Baseline (no randomization, source)
    source_env = gym.make("CustomHopper-source-v0")
    target_env = gym.make("CustomHopper-target-v0")

    print("\n▶ Training baseline (source→source)...")
    m1, s1 = PPO_agent(source_env, source_env, "ppo_baseline_source")

    print("\n Testing baseline (source→target)...")
    model = PPO.load(os.path.join(LOG_DIR, "ppo_baseline_source"))
    m2, s2 = evaluate_policy(model, target_env, n_eval_episodes=20)

    # ➤ UDR (uniform domain randomization)
    print("\n Training UDR (source)...")
    udr_env = gym.make("CustomHopper-source-v0")
    udr_env.set_random_parameters = udr_env.set_random_parameters  # ensure DR is active

    def reset_with_udr():
        udr_env.set_random_parameters()
        return udr_env.reset()
    udr_env.reset = reset_with_udr

    m3, s3 = PPO_agent(udr_env, target_env, "ppo_udr")

    # ➤ CDR (curriculum domain randomization)
    print("\n Training CDR (source)...")
    cdr_env = gym.make("CustomHopper-source-v0")
    
    def cdr_reset():
        cdr_env.cdr_step += 1
        progress = min(1.0, cdr_env.cdr_step / cdr_env.cdr_max_steps)
        r = cdr_env.min_range + (cdr_env.max_range - cdr_env.min_range) * progress
        cdr_env.mass_ranges = {
            "thigh": (cdr_env.original_masses[1] * (1 - r), cdr_env.original_masses[1] * (1 + r)),
            "leg": (cdr_env.original_masses[2] * (1 - r), cdr_env.original_masses[2] * (1 + r)),
            "foot": (cdr_env.original_masses[3] * (1 - r), cdr_env.original_masses[3] * (1 + r)),
        }
        cdr_env.set_random_parameters()
        return cdr_env.reset()
    cdr_env.reset = cdr_reset

    m4, s4 = PPO_agent(cdr_env, target_env, "ppo_cdr")

    
    print("\n RISULTATI FINALI")
    print(f"Baseline (source→source): {m1:.2f} ± {s1:.2f}")
    print(f"Baseline (source→target): {m2:.2f} ± {s2:.2f}")
    print(f"UDR       (source→target): {m3:.2f} ± {s3:.2f}")
    print(f"CDR       (source→target): {m4:.2f} ± {s4:.2f}")


if __name__ == "__main__":
    train_and_compare()


