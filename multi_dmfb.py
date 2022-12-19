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
	NON = 4

class MEDAEnv:
	def __init__(self, w=8, h=8, s_modules=0, d_modules=0, test_flag=False):
		super(MEDAEnv, self).__init__()
		assert w > 0 and h > 0
		self.w = w
		self.h = h
	
		self.actions = Actions
		self.action_space = len(self.actions)
		self.observation_space = w*h
		self.rewards = [0.]*2
		self.dones = [False]*2
		self.states = [[]]*2
		self.states[0] = [0,0]
		self.states[1] = [w-1,h-1]

		self.n_steps = 0
		self.n_max_steps = 5*(w+h)

		self.map_symbols = Symbols()
		self.mapclass = MakeMap(w=self.w,h=self.h, dsize=1,s_modules=s_modules,d_modules=d_modules)
		self.map = self.mapclass.gen_random_map()

		self.test_flag = test_flag
		self.dynamic_flag = 0

	def reset(self, test_map=None):
		self.rewards = [0.]*2
		self.dones = [False]*2
		self.states = [[]]*2
		self.states[0] = [0,0]
		self.states[1] = [self.w-1,self.h-1]

		self.n_steps = 0
		self.dynamic_flag = 0

		if self.test_flag == False:
			self.map = self.mapclass.gen_random_map()
		else:
			self.map = test_map

		obs = self.get_all_obs()

		return obs

	def step(self, actions):
		message = None
		self.n_steps += 1
		self.dones = [False]*2

		self.update_positions(actions)
		if self.dones[0]==True and self.dones[1]==True:
			self.dones = [True]*2
		else:
			self.dones = [False]*2

		obs = self.get_all_obs()

		return obs, self.rewards, self.dones, {}

	def update_positions(self, actions):
		for i in range(2):
			state_ = list(self.states[i])

			if actions[i] == Actions.U:
				state_[1] -= 1
			elif actions[i] == Actions.R:
				state_[0] += 1
			elif actions[i] == Actions.D:
				state_[1] += 1
			elif actions[i] == Actions.L:
				state_[0] -= 1
			elif actions[i] == Actions.NON:
				state_ = self.states[i]
			else:
				print("Unexpected action")
				return 0

			if 0>state_[0] or 0>state_[1] or state_[0]>self.w-1 or state_[1]>self.h-1:
				self.dynamic_flag = 0
			elif self.map[state_[1]][state_[0]] == self.map_symbols.Static_module:
				self.dynamic_flag = 0
			elif self.map[state_[1]][state_[0]] == self.map_symbols.Dynamic_module:
				self.map[state_[1]][state_[0]] = self.map_symbols.Static_module
				self.dynamic_flag = 1
			else:
				self.map[self.states[i][1]][self.states[i][0]] = self.map_symbols.Health
				self.states[i] = state_
				self.map[self.states[i][1]][self.states[i][0]] = self.map_symbols.State
			if self.states[0] == self.states[1]:
				self.rewards[i] = 0
				self.dones[i] = True
			elif self.n_steps == self.n_max_steps:
				self.rewards[i] = -1.0
				self.dones[i] = True
			elif self.dynamic_flag == 1:
				self.rewards[i] = 0
				self.dynamic_flag = 0
				message = "derror"
			else:
				self.rewards[i] = -0.1


	def get_all_obs(self):
		obs = [[]]*2
		obs[0] = self.get_obs(self.states[0], self.states[1])
		obs[1] = self.get_obs(self.states[1], self.states[0])

		obs[0] = np.reshape(obs[0], -1)
		obs[1] = np.reshape(obs[1], -1)
		return obs

	def get_obs(self, main, other):
		obs = np.zeros(shape = (self.w, self.h))
		for i in range(self.w):
			for j in range(self.h):
				if self.map[j][i] == self.map_symbols.Static_module:
					obs[i][j] = 1
#		print("main",main)
#		print("other",other)
		obs[main] = 2
		obs[other] = 3

		return obs
