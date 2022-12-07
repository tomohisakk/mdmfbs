import numpy as np
from agent import Agent
from vec_env import make_vec_envs
from multi_dmfb import MEDAEnv

from tensorboardX import SummaryWriter

def clip_reward(x):
	rewards = []
	for r in x:
		if r < -1:
			rewards.append(-1.0)
		elif r > 1:
			rewards.append(1.0)
		else:
			rewards.append(r)
	return rewards

if __name__ == '__main__':
	# env_id = 'LunarLander-v2'
	#env_id = 'CartPole-v0'
	writer = SummaryWriter()
	random_seed = 0
	n_procs = 2
#	env = make_vec_envs(env_id, random_seed, n_procs)
	env = MEDAEnv()
	N = 1024
	batch_size = 32
	n_epochs = 10
	alpha = 1E-4
	n_actions = env.action_space
	agent = Agent(n_actions=n_actions, batch_size=batch_size,
				alpha=alpha, n_epochs=n_epochs, T=N,
				input_dims=env.observation_space,
				n_procs=n_procs, writer=writer)

	score_history = []
	max_steps = 1_000_000
	total_steps = 0
	traj_length = 0
	episode = 1

	while total_steps < max_steps:
		observation = env.reset()
		done = [False] * n_procs
		score = [0] * n_procs
		scores = 0
		while not any(done):
			action, prob = agent.choose_action(observation)
			observation_, reward, done, info = env.step(action)
			r = clip_reward(reward)
			total_steps += 1
			traj_length += 1
			score += reward
			mask = [0.0 if d else 1.0 for d in done]
			agent.remember(observation, observation_, action, prob, r, mask)
			if traj_length % N == 0:
				agent.learn(episode)
				traj_length = 0
			observation = observation_
		
		scores += sum(score)


		score_history.append(scores)
		avg_score = np.mean(score_history[-100:])
		writer.add_scalar("avg_scores", avg_score, episode)

		print('Episode {} total steps {} avg score {:.1f}'.format(episode, total_steps, avg_score))
		episode += 1
	env.close()