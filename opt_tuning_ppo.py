
#ottimizzazione parametri con optuna
import optuna
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
import pandas as pd
from env.custom_hopper import *
def objective(trial):
   #in base ai parametri da ottimizzare si usa suggest_float\int o categorical (per una lista fissata dall'inizio)
    # IPERPARAMETRI PRINCIPALI (range ristretti basati su letteratura) ->Bayesian Optimization with Informed Priors" per report
    #"domain knowledge"  "expert knowledge" "warm start"
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 8e-4, log=True)
    gamma = trial.suggest_float('gamma', 0.99, 0.9999)
    ent_coef = trial.suggest_float('ent_coef', 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128, 256])
    n_steps = trial.suggest_categorical('n_steps', [1024, 2048, 4096])
    n_epochs = trial.suggest_int('n_epochs', 5, 20)
    clip_range = trial.suggest_float('clip_range', 0.15, 0.3)
    gae_lambda = trial.suggest_float('gae_lambda', 0.9, 0.99)
    source_env = gym.make('CustomHopper-source-v0')
    source_udr_env = gym.make('CustomHopper-source-udr-v0')
    #qui si crea un modello identico al training che viene addestrato con combinazioni diverse di parametri ogni volta
    #quindi si usa normalize advantage e TUTTI I PARAMETRI DEL TRAIN (con quelli non da ottimizzare fissi, per avere risultati comparabili 
    #e utili per training vero e proprio)
    model = PPO(
        "MlpPolicy",
        source_udr_env,
        # Iperparametri di apprendimento
        learning_rate=learning_rate,
        gamma=gamma,
        ent_coef=ent_coef,
        # Parametri di raccolta dati
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        # Parametri specifici di PPO
        clip_range=clip_range,
        gae_lambda=gae_lambda,
        normalize_advantage=True, #fisso
        #tensorboard_log="./ppo_tensorboard/" eventualmente da integrare
    )
    model.learn(total_timesteps=100000)
    
    #valutazione degli iperparametri di questa objective
    # fatta direttamente source ->target perche caso piu complicato e che si userà anche dopo per UDR CDR
    target_env = gym.make('CustomHopper-target-v0')
    #eventualmente qui si puo integrare tensorboard e valutare altre cose oltre standard reward e mean reward
    mean_reward, std_reward = evaluate_policy(
        model, target_env, n_eval_episodes=50, deterministic=True
    )

    source_udr_env.close()
    target_env.close()
    del model
    #metrica per tenere conto del fatto che la varianza possa incidere e il mean reward da solo non è sufficiente ma
    #moltiplicata per un fattore per tenere conto del fatto che se penalizziamo troppo la std sul training c'è bias
    return (mean_reward-0.1*std_reward)
    #eliminiamo il modello perche non ci interessa interessano solo i risultati sugli iperparametri
    

def run_optimization():
    # CREAZIONE DELLO STUDY (il gestore dei vari esperimenti )
    #volendo lo study si puo salvare in memoria in sql
    study = optuna.create_study(
        direction='maximize',
        
        # sampler: algoritmo di ottimizzazione
        # TPESampler è il default e più versatile (sembra un random tree bayesiano, va a campionare prossime combinazioni su cui fare esperimenti
        #prendendo quelli che hanno dato risultati migliori con probabilità piu alta(a meno di pruning che conviene 
        #implementare per ragioni computazionali
        sampler=optuna.samplers.TPESampler(seed=2025),

        pruner=None, #se non si specifica viene usato un pruner basato sulla mediana
    )
    print("Inizio ottimizzazione...")
    study.optimize(objective, 
        n_trials =10, #di prova, essendo veloce si possono fare almeno 10 trial (prove con random seed diversi del processo di ottimizzazione)
        show_progress_bar=True ) 
    
    # ANALISI DEI RISULTATI
    print("\n=== RISULTATI ===")
    print(f"Miglior valore: {study.best_value:.2f}")
    print("Migliori parametri:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    
    df = study.trials_dataframe()
    df.to_csv('optimization_results_udr.csv')
    
    return study
run_optimization()
