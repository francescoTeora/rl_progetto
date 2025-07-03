
#ottimizzazione parametri con optuna
import optuna
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
import pandas as pd
from env.custom_hopper import *
def objective(trial):
   #in base ai parametri da ottimizzare si usa suggest_float\int o categorical (per una lista fissata dall'inizio)
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
    gamma = trial.suggest_float('gamma', 0.95, 0.9999)
    ent_coef = trial.suggest_float('ent_coef', 1e-6, 1e-2, log=True)
    source_env = gym.make('CustomHopper-source-v0')
    #qui si crea un modello identico al training che viene addestrato con combinazioni diverse di parametri ogni volta
    #quindi si usa normalize advantage e TUTTI I PARAMETRI DEL TRAIN (con quelli non da ottimizzare fissi, per avere risultati comparabili 
    #e utili per training vero e proprio)
    model = PPO(
        "MlpPolicy",
        source_env,
        learning_rate=learning_rate,
        gamma=gamma,
        #n_steps=n_step da aggiungere numero di step come iperparametro da ottimizzare eventualmente
        ent_coef=ent_coef,
        normalize_advantage=True, #fisso
        #tensorboard_log="./ppo_tensorboard/" eventualmente da integrare
    )
    model.learn(total_timesteps=20000)
    
    #valutazione degli iperparametri di questa objective
    # fatta direttamente source ->target perche caso piu complicato e che si userà anche dopo per UDR CDR
    target_env = gym.make('CustomHopper-target-v0')
    #eventualmente qui si puo integrare tensorboard e valutare altre cose oltre standard reward e mean reward
    mean_reward, std_reward = evaluate_policy(
        model, target_env, n_eval_episodes=50, deterministic=True
    )

    source_env.close()
    target_env.close()
    del model  #eliminiamo il modello perche non ci interessa interessano solo i risultati sugli iperparametri
    return mean_reward

def run_optimization():
    # CREAZIONE DELLO STUDY (il gestore dei vari esperimenti )
    #volendo lo study si puo salvare in memoria in sql
    study = optuna.create_study(
        direction='maximize',
        
        # sampler: algoritmo di ottimizzazione
        # TPESampler è il default e più versatile (sembra un random tree bayesiano, va a campionare prossime combinazioni su cui fare esperimenti
        #prendendo quelli che hanno dato risultati migliori con probabilità piu alta(a meno di pruning che conviene 
        #implementare per ragioni computazionali
        sampler=optuna.samplers.TPESampler(seed=1648),

        pruner=None, #se non si specifica viene usato un pruner basato sulla mediana
    )
    print("Inizio ottimizzazione...")
    study.optimize(objective, 
        n_trials =2, #eventualmente di piu se è veloce 
        show_progress_bar=True ) 
    
    # ANALISI DEI RISULTATI
    print("\n=== RISULTATI ===")
    print(f"Miglior valore: {study.best_value:.2f}")
    print("Migliori parametri:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    
    df = study.trials_dataframe()
    df.to_csv('optimization_results.csv')
    
    return study
run_optimization()