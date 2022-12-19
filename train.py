import numpy as np
from agent import Agent
from multi_dmfb import MEDAEnv

from tensorboardX import SummaryWriter

if __name__ == '__main__':
	writer = SummaryWriter(log_dir="runs/tmp")
	random_seed = 0
	n_procs = 2
	env = MEDAEnv()
	N = 2048
	batch_size = 32
	n_epochs = 3
	alpha = 1E-4
	env_name="tmp"
	n_actions = env.action_space
	agent = Agent(n_actions=n_actions, batch_size=batch_size,
				alpha=alpha, n_epochs=n_epochs, T=N,
				input_dims=env.observation_space,
				n_procs=2, writer=writer)

	score_history = []
	max_steps = 1_000_000
	total_steps = 0
	traj_length = 0
	episode = 1

	while True:
		observation = env.reset()
		done = [False] * 2
		score = [0]*2
		scores = 0
		while not any(done):
#			print(observation)
			action, prob = agent.choose_action(observation)
			observation_, reward, done, info = env.step(action)
#			r = clip_reward(reward)
			traj_length += 1
			score += reward
			mask = [0.0 if d else 1.0 for d in done]
			agent.remember(observation, observation_, action, prob, reward, mask)
			if traj_length % N == 0:
				agent.learn(episode)
				traj_length = 0
			observation = observation_
		
		scores += sum(score)
		score_history.append(scores)
		avg_score = np.mean(score_history[-100:])
		writer.add_scalar("avg_scores", avg_score, episode)
		if episode%100==0:
			print('Episode {} avg score {:.1f}'.format(episode, avg_score))
		episode += 1
	env.close()
