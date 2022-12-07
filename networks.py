import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical


class DiscreteActorNetwork(nn.Module):
	def __init__(self, n_actions, input_dims, alpha,
				fc1_dims=128, fc2_dims=128, chkpt_dir='models/'):
		super(DiscreteActorNetwork, self).__init__()
		self.checkpoint_file = os.path.join(chkpt_dir,
											'actor_discrete_ppo')
		self.fc1 = nn.Linear(input_dims, fc1_dims)
		self.fc2 = nn.Linear(fc1_dims, fc2_dims)
		self.pi = nn.Linear(fc2_dims, n_actions)

		self.optimizer = optim.Adam(self.parameters(), lr=alpha)
		self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
		self.to(self.device)

	def forward(self, state):
		x = T.tanh(self.fc1(state))
		x = T.tanh(self.fc2(x))
		pi = F.softmax(self.pi(x), dim=1)
		dist = Categorical(pi)
		return dist

	def save_checkpoint(self):
		T.save(self.state_dict(), self.checkpoint_file)

	def load_checkpoint(self):
		self.load_state_dict(T.load(self.checkpoint_file))


class DiscreteCriticNetwork(nn.Module):
	def __init__(self, input_dims, alpha,
				fc1_dims=128, fc2_dims=128, chkpt_dir='models/'):
		super(DiscreteCriticNetwork, self).__init__()
		self.checkpoint_file = os.path.join(chkpt_dir,
											'critic_discrete_ppo')
		self.fc1 = nn.Linear(input_dims, fc1_dims)
		self.fc2 = nn.Linear(fc1_dims, fc2_dims)
		self.v = nn.Linear(fc2_dims, 1)

		self.optimizer = optim.Adam(self.parameters(), lr=alpha)
		self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
		self.to(self.device)

	def forward(self, state):
		x = T.tanh(self.fc1(state))
		x = T.tanh(self.fc2(x))
		v = self.v(x)

		return v

	def save_checkpoint(self):
		T.save(self.state_dict(), self.checkpoint_file)

	def load_checkpoint(self):
		self.load_state_dict(T.load(self.checkpoint_file))
