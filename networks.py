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

class PPO(nn.Module):
	def __init__(self, input_shape, n_actions):
		super(PPO, self).__init__()

		self.conv = nn.Sequential(
			nn.Conv2d(input_shape[0], 32, kernel_size=3, stride=1),
			nn.ReLU(),
			nn.Conv2d(32, 64, kernel_size=3, stride=1),
			nn.ReLU(),
			nn.Conv2d(64, 64, kernel_size=3, stride=1),
			nn.ReLU()
		)

		conv_out_size = self._get_conv_out(input_shape)
		hidden_size = 256
#		print(hidden_size)

		self.actor = nn.Sequential(
			nn.Linear(conv_out_size, int(hidden_size)),
			nn.ReLU(),
			nn.Linear(int(hidden_size), n_actions)
		)
		self.critic = nn.Sequential(
			nn.Linear(conv_out_size, int(hidden_size)),
			nn.ReLU(),
			nn.Linear(int(hidden_size), 1)
		)

	def _get_conv_out(self, shape):
		o = self.conv(T.zeros(1, *shape))
		return int(np.prod(o.size()))

	def forward(self, x):
#		fx = x.float() / 1
		conv_out = self.conv(x).view(x.size()[0], -1)
		return self.actor(conv_out), self.critic(conv_out)

	def save_checkpoint(self, env_name):
		print("... saveing checkpoint ...")
		T.save(self.state_dict(), "saves/" + env_name + ".pt")

	def load_checkpoint(self, env_name):
		self.load_state_dict(T.load("saves/" + env_name + ".pt"))
