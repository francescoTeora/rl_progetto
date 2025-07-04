"""Train an RL agent on the OpenAI Gym Hopper environment using REINFORCE and Actor-critic algorithms """
import argparse
import torch
import gym
from env.custom_hopper import *
from agent import Agent, Policy
import os
import re

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-episodes', default=100000, type=int, help='Number of training episodes')
    parser.add_argument('--print-every', default=1000, type=int, help='Print info every <> episodes')
    parser.add_argument('--device', default='cuda', type=str, help='network device [cpu, cuda]')
    parser.add_argument('--algorithm', default='actor_critic', type=str, choices=['reinforce', 'reinforce_baseline', 'actor_critic'], 
                       help='Algorithm to use: reinforce, reinforce_baseline, or actor_critic')
    return parser.parse_args()

args = parse_args()

def get_model_paths(algorithm):
    """
    Restituisce i percorsi delle cartelle e dei file per l'algoritmo specificato
    """
    if algorithm == 'reinforce':
        checkpoint_dir = 'checkpoints/reinforce'
        model_dir = 'models/reinforce'
        model_name = 'reinforce_model.mdl'
    elif algorithm == 'reinforce_baseline':
        checkpoint_dir = 'checkpoints/reinforce_baseline'
        model_dir = 'models/reinforce_baseline'
        model_name = 'reinforce_baseline_model.mdl'
    elif algorithm == 'actor_critic':
        checkpoint_dir = 'checkpoints/actor_critic'
        model_dir = 'models/actor_critic'
        model_name = 'actor_critic_model.mdl'
    else:
        raise ValueError(f"Algorithm not valid: {algorithm}")
    
    # Crea le cartelle se non esistono
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    return checkpoint_dir, model_dir, model_name

def find_latest_checkpoint(checkpoint_dir):
    """
    Trova il checkpoint più recente nella cartella specificata
    """
    if not os.path.exists(checkpoint_dir):
        return None, 0
    
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.mdl')]
    if not checkpoint_files:
        return None, 0
    
    # Trova il checkpoint con il numero di episodio più alto
    latest_episode = 0
    latest_file = None
    
    for file in checkpoint_files:
        match = re.search(r'model_ep(\d+)\.mdl', file)
        if match:
            episode = int(match.group(1))
            if episode > latest_episode:
                latest_episode = episode
                latest_file = file
    
    if latest_file:
        return os.path.join(checkpoint_dir, latest_file), latest_episode
    
    return None, 0

def main():
    env = gym.make('CustomHopper-source-v0')
    env = gym.make('CustomHopper-target-v0')

    print('Action space:', env.action_space)
    print('State space:', env.observation_space)
    print('Dynamics parameters:', env.get_parameters())
    print(f'Algorithm: {args.algorithm}')

    """
    Training
    """
    observation_space_dim = env.observation_space.shape[-1]
    action_space_dim = env.action_space.shape[-1]
    #folder where to find checkpoints
    checkpoint_dir, model_dir, model_name = get_model_paths(args.algorithm)
    
    #find most recent checkpoint
    checkpoint_path, start_episode = find_latest_checkpoint(checkpoint_dir)
    
    #configure the agent with respect to the algorithm
    policy = Policy(observation_space_dim, action_space_dim)
    if args.algorithm == 'reinforce':
        agent = Agent(policy, use_baseline=False, critic=False, device=args.device)
    elif args.algorithm == 'reinforce_baseline':
        agent = Agent(policy, use_baseline=True, critic=False, device=args.device)
    elif args.algorithm == 'actor_critic':
        agent = Agent(policy, use_baseline=False, critic=True, device=args.device)
    
    batch_size = 30
    
    # Carica il checkpoint se esiste
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"TROVATO CHECKPOINT E RIPRESO TRAINING DA {checkpoint_path}")
        policy.load_state_dict(torch.load(checkpoint_path)['model_state_dict'], strict=True)
        print(f"Ripartendo dall'episodio {start_episode}")
    else:
        print("Nessun checkpoint trovato, training da 0")

    # Training loop
    for episode in range(start_episode, args.n_episodes):
        done = False
        train_reward = 0
        state = env.reset()  # Reset the environment and observe the initial state
        
        while not done:  # Loop until the episode is over
            action, action_probabilities = agent.get_action(state)
            previous_state = state
            state, reward, done, info = env.step(action.detach().cpu().numpy())
            agent.store_outcome(previous_state, state, action_probabilities, reward, done)
            train_reward += reward
            
        if len(agent.rewards) > 0 and episode % batch_size == 0:
            agent.update_policy()
            
        if (episode + 1) % args.print_every == 0:
            print(f'Training episode: {episode}')
            print(f'Episode return: {train_reward}')
            
            # Salva il checkpoint nella cartella specifica dell'algoritmo
            checkpoint_file = os.path.join(checkpoint_dir, f"model_ep{episode + 1}.mdl")
            torch.save({
                'model_state_dict': agent.policy.state_dict(),
                'episode': episode + 1,
                'algorithm': args.algorithm
            }, checkpoint_file)
            print(f"Checkpoint salvato: {checkpoint_file}")

    # Salva il modello finale nella cartella specifica dell'algoritmo
    final_model_path = os.path.join(model_dir, model_name)
    torch.save(agent.policy.state_dict(), final_model_path)
    print(f"Modello finale salvato: {final_model_path}")

if __name__ == '__main__':
    main()