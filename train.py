"""Train an RL agent on the OpenAI Gym Hopper environment using
    REINFORCE and Actor-critic algorithms
"""
import argparse

import torch
import gym

from env.custom_hopper import *
from agent import Agent, Policy
import os
import re
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-episodes', default=20000, type=int, help='Number of training episodes')
    parser.add_argument('--print-every', default=100, type=int, help='Print info every <> episodes')
    parser.add_argument('--device', default='cuda', type=str, help='network device [cpu, cuda]')

    return parser.parse_args()

args = parse_args()


def main():

	env = gym.make('CustomHopper-source-v0')
	# env = gym.make('CustomHopper-target-v0')

	print('Action space:', env.action_space)
	print('State space:', env.observation_space)
	print('Dynamics parameters:', env.get_parameters())


	"""
		Training
	"""
	observation_space_dim = env.observation_space.shape[-1]
	action_space_dim = env.action_space.shape[-1]
	"""
	in questa parte del codice si carica un eventuale checkpoint
	Possibili modifiche, specificare nomi modello diversi per ogni task che si va a fare e di conseguenza creare piu cartelle checkpoint
	sempre con nomi diversi in base al modello a cui si riferiscono
	"""
	
	checkpoint_path = 'checkpoints/model_ep20200.mdl'  
	start_episode = 0
	policy = Policy(observation_space_dim, action_space_dim)
	agent = Agent(policy, device=args.device)
	batch_size = 10
	if os.path.exists(checkpoint_path):
		print(f"TROVATO CHECKPOINT E RIPRESO TRAINING DA Lì {checkpoint_path}")
		policy.load_state_dict(torch.load(checkpoint_path)['model_state_dict'], strict=True)
		match = re.search(r'model_ep(\d+)\.mdl', checkpoint_path)
		if match:
			start_episode = int(match.group(1))
		else:
			print("checkpoint non valido")
	else:
		print("Nessun checkpoint trovato, training da 0")

    #
    # TASK 2 and 3: interleave data collection to policy updates
    #
	####[##################]SE VOGLIAMO CARICARE UN CHECKPOINT, DOPO AVER SCELTO DA QUALE MODELLO E CHE NUMERO EPISODIO MODIFICARE START EPISODE######
	#si potrebbe fare in modo automatico (modifica)
	

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
		if len(agent.rewards) > 0 and episode%batch_size==0:
			agent.update_policy()
		if (episode+1)%args.print_every == 0:
			print('Training episode:', episode)
			print('Episode return:', train_reward)
			"""
			qui si va a salvare il modello ogni x episodi, possibili modifiche farlo con un numero diverso rispetto ai print, e quindi magari
			ogni 1000 episodi e poi si puo riprendere da lì
			"""
			torch.save({
				'model_state_dict': agent.policy.state_dict(),
			}, f"checkpoints/model_ep{episode + 1}.mdl")




	torch.save(agent.policy.state_dict(), "model.mdl")
	
if __name__ == '__main__':
	main()