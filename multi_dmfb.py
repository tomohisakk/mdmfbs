from enum import IntEnum
import math
import random
import copy
import numpy as np

from map import MakeMap
from map import Symbols

class Actions(IntEnum):
	U = 0
	R = 1
	D = 2
	L = 3

class Droplet:
	def __init__(self, x, y):
		self.x = x
		self.y = y

	def is_overlaping(self, another):
		if self.x == another.x and self.y == another.y:
			return True
		else:
			return False

	def is_too_close(self, another):
		distance = self.get_dist(another)
		if distance <= 1:
			return True
		else:
			return False

	def get_dist(self, another):
		diff_x = self.x - another.x
		diff_y = self.y - another.y
		return math.sqrt(diff_x*diff_x + diff_y*diff_y)

	def update(self, action, w, h):
		state_ = [self.x, self.y]

		if action == Actions.U:
			state_[1] -= 1
		elif action == Actions.R:
			state_[0] += 1
		elif action == Actions.D:
			state_[1] += 1
		elif action == Actions.L:
			state_[0] -= 1
		else:
			print("Unexpected action")
			return 0
		
		if state_[0] < 0 or state_[1] < 0 or state_[0] > w-1 or state_[1] > h-1:
			return

		self.x, self.y = state_

class Routing:
	def __init__(self, w, h, n_agents):
		self.w = w
		self.h = h
		self.n_agents = n_agents

		self.starts = []
		self.droplets = []
		self.goals = []
		self.dists = []

		for i in range(n_agents):
			self.add_task()

		self.n_steps = [0] * self.n_agents

	def add_task(self):
		self._gen_legal_droplet(self.droplets)
		self._gen_legal_droplet(self.goals)

		while(self.droplets[-1].is_overlaping(self.goals[-1])):
			self.goals.pop()
			self._gen_legal_droplet(self.goals)
		self.dists.append(self.droplets[-1].get_dist(self.goals[-1]))
		self.starts.append(copy.deepcopy(self.droplets[-1]))

	def _gen_legal_droplet(self, dtype):
		state = (random.randint(0, self.w-1), random.randint(0, self.h-1))
		new_droplet = Droplet(state[1], state[0])
		while not self._is_good_droplet(new_droplet, dtype):
			state = (random.randint(0, self.w-1), random.randint(0, self.h-1))
			new_droplet = Droplet(state[1], state[0])
		dtype.append(new_droplet)

	def _is_good_droplet(self, new_d, dtype):
		for d in dtype:
			if d.is_too_close(new_d):
				return False
		return True

	def move_droplets(self, actions):
		rewards = []
		for i in range(self.n_agents):
			rewards.append(self.move_one_droplet(i, actions[i]))
		return rewards

	def move_one_droplet(self, i, action):
		if self.dists[i] == 0:
			self.states[i] = copyy.deepcopy(self.goal[i])
			self.dists[i] = 0
			reward = 0.0
		else:
			self.droplets[i].update(action, self.w, self.h)
			if self.droplets[i]==(self.w-1, self.h-1):
				reward = 1.0
#			elif dist_ < self.dists[i]:
#				reward = 0.5
			else:
				reward = -0.1

		return reward

	def refresh(self):
		self.starts.clear()
		self.droplets.clear()
		self.goals.clear()
		self.dists.clear()
		for i in range(self.n_agents):
			self.add_task()

	def is_done(self):
		return [dist == 0 for dist in self.dists]

class MEDAEnv:
	def __init__(self, w=8, h=8, n_agents=2, s_modules=0, d_modules=0, test_flag=False):
		super(MEDAEnv, self).__init__()
		assert w > 0 and h > 0 and n_agents > 0
		self.w = w
		self.h = h
		self.n_agents = n_agents
		self.agents = [i for i in range(n_agents)]

		self.actions = Actions
		self.action_space = len(self.actions)
		self.observation_space = w*h
		self.rewards = [0.]*n_agents
		self.dones = [False]*n_agents

		self.routing = Routing(w, h, n_agents)

		self.n_steps = 0
		self.n_max_steps = 30

		self.mapclass = MakeMap(w=self.w,h=self.h, dsize=1,s_modules=s_modules,d_modules=d_modules)
		self.map = self.mapclass.gen_random_map()

	def reset(self):
		self.rewards = [0.]*self.n_agents
		self.dones = [False]*self.n_agents

		self.n_steps = 0
		self.dynamic_flag = 0

		self.routing.refresh()
		obs = self.get_obs()

		return obs

	def step(self, actions):
		message = None
		self.n_steps += 1

		rewards = self.routing.move_droplets(actions)
		for i, r in zip(self.agents, rewards):
			self.rewards[i] = r

		obs = self.get_obs()

		if self.n_steps < self.n_max_steps:
			is_dones = self.routing.is_done()
			for key, s in zip(self.agents, is_dones):
				self.dones[key] = s
		else:
			for key in self.agents:
				self.dones[key] = True

		return obs, self.rewards, self.dones, {}

	def get_obs(self):
		obs = [[]]*self.n_agents
		for i, agent in enumerate(self.agents):
			obs[i] = self.get_one_obs(i)
#		print(obs)
		obs[0] = np.reshape(obs[0], -1)
		obs[1] = np.reshape(obs[1], -1)

		return obs

	def get_one_obs(self, agent_index):
		obs = np.zeros((self.h, self.w))

		for i in range(self.routing.n_agents):
			if i == agent_index:
				continue
			degrade = self.routing.droplets[i]
			obs = self._make_obs(obs, degrade, 1)

		goal = self.routing.goals[agent_index]
		obs = self._make_obs(obs, goal, 2)

		state = self.routing.droplets[agent_index]
		obs = self._make_obs(obs, state, 3)

		#print("--- Obs ---")
		#print(obs)

		print(self.map)
		return obs

	def _make_obs(self, obs, droplet, status):
		x = 0 if droplet.x < 0 else droplet.x
		x = self.w-1 if droplet.x >= self.w else droplet.x

		y = 0 if droplet.y < 0 else droplet.y
		y = self.h-1 if droplet.y >= self.h else droplet.y

		obs[y][x] = status

		return obs

"""
N_AGENTS = 2
N_GAMES = 100

env = MEDAEnv(w=10, l=10, n_agents=N_AGENTS)

for i in range(N_GAMES):

	scores = 0
	n_steps = 0
	env.reset(n_modules=0)
	dones = [False]*N_AGENTS

	while not any(dones):
		n_steps += 1
		a = [random.randint(0,3)]*N_AGENTS
		print("--- Actions ---")
		print(a)
		print()
		obs_, rewards, dones, _ = env.step(a)
		scores += sum(rewards)

		print("--- Observation ---")
		print(obs_)
		print()
		

		print("--- Reward ---")
		print(rewards)
		print()

		print("--- Dones ---")
		print(dones)
		print()


#	print("--- Game end ---")
	print("Total score")
	print(scores)

	print("--- N_steps ---")
	print(n_steps)
	
	print()
"""
