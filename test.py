"""Test an RL agent on the OpenAI Gym Hopper environment"""
import argparse
import numpy as np
import torch
import gym
import matplotlib.pyplot as plt

from env.custom_hopper import *
from agent import Agent, Policy

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default="/root/projectMl/rl_mldl_25/models/reinforce_baseline_model.mdl", type=str, help='Model path')
    parser.add_argument('--device', default='cuda', type=str, help='network device [cpu, cuda]')
    parser.add_argument('--render', default=False, action='store_true', help='Render the simulator')
    parser.add_argument('--episodes', default=100, type=int, help='Number of test episodes')

    return parser.parse_args()

args = parse_args()


def main():
    # Define the two models to compare
    models = {
        'REINFORCE': '/root/projectMl/rl_mldl_25/models/reinforce_model.mdl',
        'REINFORCE Baseline': '/root/projectMl/rl_mldl_25/models/reinforce_baseline_model.mdl'
    }
    
    results = {}
    
    for model_name, model_path in models.items():
        print(f"\nTesting {model_name}...")
        
        env = gym.make('CustomHopper-target-v0')

        print('Action space:', env.action_space)
        print('State space:', env.observation_space)
        print('Dynamics parameters:', env.get_parameters())
        
        observation_space_dim = env.observation_space.shape[-1]
        action_space_dim = env.action_space.shape[-1]

        policy = Policy(observation_space_dim, action_space_dim)
        policy.load_state_dict(torch.load(model_path), strict=True)

        agent = Agent(policy, device=args.device)
        episode_rewards = np.zeros(args.episodes)
        
        for episode in range(args.episodes):
            done = False
            test_reward = 0
            state = env.reset()

            while not done:
                action, _ = agent.get_action(state, evaluation=True)
                state, reward, done, info = env.step(action.detach().cpu().numpy())

                if args.render:
                    env.render()

                test_reward += reward
            
            episode_rewards[episode] = test_reward
            print(f"Episode: {episode} | Return: {test_reward}")
        
        mean_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)
        
        results[model_name] = {
            'mean': mean_reward,
            'std': std_reward,
            'rewards': episode_rewards
        }
        
        print(f"{model_name}--------------------->{mean_reward}±{std_reward}")
        env.close()
    
    # Create comparison plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: Average Returns Comparison
    model_names = list(results.keys())
    means = [results[name]['mean'] for name in model_names]
    stds = [results[name]['std'] for name in model_names]
    
    ax1.bar(model_names, means, alpha=0.7, color=['skyblue', 'lightgreen'])
    ax1.set_title('Average Returns Comparison')
    ax1.set_ylabel('Average Return')
    ax1.set_xlabel('Algorithm')
    ax1.tick_params(axis='x', rotation=45)
    
    # Plot 2: Standard Deviation Comparison
    ax2.bar(model_names, stds, alpha=0.7, color=['skyblue', 'lightgreen'])
    ax2.set_title('Standard Deviation Comparison')
    ax2.set_ylabel('Standard Deviation')
    ax2.set_xlabel('Algorithm')
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('algorithm_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print summary statistics
    print("\n" + "="*60)
    print("SUMMARY COMPARISON")
    print("="*60)
    for model_name, result in results.items():
        print(f"{model_name}: {result['mean']:.2f} ± {result['std']:.2f}")

if __name__ == '__main__':
    main()