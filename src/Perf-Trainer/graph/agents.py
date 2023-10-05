import os
import os.path as path
import csv
import json
import time
import datetime

import numpy as np
import pandas as pd

import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.data

# import graph.train_utils as train_utils
# from graph.model import DQN_v0, ReplayMemory, DQN_AB, ReplayMemoryTime	
import train_utils as train_utils
from model import DQN_v0, ReplayMemory, DQN_AB
"""
RL Agents are responsible for train/inference
Available Agents:
1. Vanilla Agent
2. Agent with Action Branching
"""

# Vanilla Agent without action branching
class DQN_AGENT():
	def __init__(self, s_dim, a_dim, buffer_size, params):
		self.eps = 0.8
		self.actions = np.arange(a_dim)
		# Experience Replay
		self.mem = ReplayMemory(buffer_size)
		# Initi networks
		self.policy_net = DQN_v0(s_dim, a_dim)
		self.target_net = DQN_v0(s_dim, a_dim)
		self.target_net.load_state_dict(self.policy_net.state_dict())
		self.target_net.eval()
		'''
		About the optimizer of DQN
		Ref:
			- https://www.reddit.com/r/reinforcementlearning/comments/ei9p3y/using_rmsprop_over_adam/
			- https://ai.stackexchange.com/questions/12268/in-q-learning-shouldnt-the-learning-rate-change-dynamically-during-the-learnin
		Defalt learning rate is 1e-2
		For Q-Learning, it is unnecessray to use a dynamic learning rate
		'''
		self.optimizer = torch.optim.RMSprop(self.policy_net.parameters())
		self.criterion = nn.SmoothL1Loss() # Huber loss
		
	def max_action(self, state):
		with torch.no_grad():
			# Inference using policy_net
			a = self.policy_net(state).max(1)[1]
		return self.actions[a]

	def e_gready_action(self, action, eps):
		# Epsilon-Gready for exploration
		p = np.random.random()
		if isinstance(action,np.ndarray):
			return [a if p < (1-eps) else np.random.choice(self.actions) for a in action]
		else:
			return action if p < (1-eps) else np.random.choice(self.actions)

	def select_action(self, state):
		return self.e_gready_action(self.max_action(state),self.eps)

	def train(self, n_round, n_update, n_batch):
		# Train on policy_net
		losses = []
		self.target_net.train()
		train_loader = torch.utils.data.DataLoader(
			self.mem, shuffle=True, batch_size=n_batch, drop_last=True)
		length = len(train_loader.dataset)
		GAMMA = 1.0
		for i, trans in enumerate(train_loader):
			states, actions, next_states, rewards = trans
			with torch.no_grad():
				next_state_values = self.target_net(next_states).max(1)[0].detach()
				expected_state_action_values = (next_state_values*GAMMA) + rewards.float()
			# Gather action-values that have been taken
			actions = actions.long()
			state_action_values = self.policy_net(states).gather(1, actions.unsqueeze(1))
			loss = self.criterion(state_action_values, expected_state_action_values.unsqueeze(1))
			losses.append(loss.item())
			self.optimizer.zero_grad()
			loss.backward()
			if i>n_update:
				break
			# Gradient clipping to prevent exploding gradients
			# for param in self.policy_net.parameters()
			# 	param.grad.data.clamp_(-1, 1)
			self.optimizer.step()
		return losses

	def save_model(self, n_round, savepath):
		train_utils.save_checkpoint({'epoch': n_round, 'model_state_dict':self.target_net.state_dict(),
	        'optimizer_state_dict':self.optimizer.state_dict()}, savepath)

	def load_model(self, loadpath):
		if not os.path.isdir(loadpath): os.makedirs(loadpath)
		checkpoint = train_utils.load_checkpoint(loadpath)
		self.policy_net.load_state_dict(checkpoint['model_state_dict'])
		self.target_net.load_state_dict(checkpoint['model_state_dict'])
		self.target_net.eval()

	def sync_model(self):
		self.target_net.load_state_dict(self.policy_net.state_dict())

# Agent with action branching without time context
class DQN_AGENT_AB():
	def __init__(self, s_dim, h_dim, branches, buffer_size, params):
		self.eps = 0.8
		# 2D action space
		self.actions = [np.arange(i) for i in branches]
		# Experience Replay(requires belief state and observations)
		self.mem = ReplayMemory(buffer_size)
		# Initi networks
		self.policy_net = DQN_AB(s_dim, h_dim, branches)
		self.target_net = DQN_AB(s_dim, h_dim, branches)
		self.target_net.load_state_dict(self.policy_net.state_dict())
		self.target_net.eval()

		self.optimizer = torch.optim.RMSprop(self.policy_net.parameters())
		self.criterion = nn.SmoothL1Loss() # Huber loss
		
	def max_action(self, state):
		# actions for multidomains
		max_actions = []
		with torch.no_grad():
			# Inference using policy_net given (domain, batch, dim)
			q_values = self.policy_net(state)
			for i in range(len(q_values)):
				domain = q_values[i].max(dim=1).indices
				max_actions.append(self.actions[i][domain])
		return max_actions

	def e_gready_action(self, actions, eps):
		# Epsilon-Gready for exploration
		final_actions = []
		for i in range(len(actions)):
			p = np.random.random()
			if isinstance(actions[i],np.ndarray):
				if p < 1- eps:
					final_actions.append(actions[i])
				else:
					# randint in (0, domain_num), for batchsize
					final_actions.append(np.random.randint(len(self.actions[i]),size=len(actions[i])))
			else:
				if p < 1- eps:
					final_actions.append(actions[i])
				else:
					final_actions.append(np.random.choice(self.actions[i]))

		return final_actions

	def select_action(self, state):
		return self.e_gready_action(self.max_action(state),self.eps)

	def train(self, n_round, n_update, n_batch):
		# Train on policy_net
		losses = []
		self.target_net.train()
		train_loader = torch.utils.data.DataLoader(
			self.mem, shuffle=True, batch_size=n_batch, drop_last=True)
		length = len(train_loader.dataset)
		GAMMA = 1.0

		# Calcuate loss for each branch and then simply sum up
		for i, trans in enumerate(train_loader):
			loss = 0.0 # initialize loss at the beginning of each batch
			states, actions, next_states, rewards = trans
			with torch.no_grad():
				target_result = self.target_net(next_states)
			policy_result = self.policy_net(states)
			# Loop through each action domain
			for j in range(len(self.actions)):
				next_state_values = target_result[j].max(dim=1)[0].detach()
				expected_state_action_values = (next_state_values*GAMMA) + rewards.float()
				# Gather action-values that have been taken
				branch_actions = actions[:,j].long()
				state_action_values = policy_result[j].gather(1, branch_actions.unsqueeze(1))
				loss += self.criterion(state_action_values, expected_state_action_values.unsqueeze(1))
			losses.append(loss.item())
			self.optimizer.zero_grad()
			loss.backward()
			if i>n_update:
				break
			# Gradient clipping to prevent exploding gradients
			# for param in self.policy_net.parameters()
			# 	param.grad.data.clamp_(-1, 1)
			self.optimizer.step()
		return losses

	def save_model(self, n_round, savepath):
		train_utils.save_checkpoint({'epoch': n_round, 'model_state_dict':self.target_net.state_dict(),
	        'optimizer_state_dict':self.optimizer.state_dict()}, savepath)

	def load_model(self, loadpath):
		if not os.path.isdir(loadpath): os.makedirs(loadpath)
		checkpoint = train_utils.load_checkpoint(loadpath)
		self.policy_net.load_state_dict(checkpoint['model_state_dict'])
		self.target_net.load_state_dict(checkpoint['model_state_dict'])
		self.target_net.eval()

	def sync_model(self):
		self.target_net.load_state_dict(self.policy_net.state_dict())


if __name__ == "__main__":
	

	N_S, N_A, N_B = 5, 3, 11

	# Test/Train Demo for DQN 
	print("Test/Train Demo for DQN ")
	agent = DQN_AGENT(N_S,N_A,N_B,None)
	
	for i in range(11):
		agent.mem.push(
			np.random.random(N_S).astype(np.float32), # state
			np.random.randint(N_A), # action
			np.random.random(N_S).astype(np.float32), # next state
			np.random.random() #reward
			)
	# Inference Test
	loader = torch.utils.data.DataLoader(agent.mem,shuffle=True,batch_size=10)
	test = None
	with torch.no_grad():
		for i, b in enumerate(loader):
			print(b.state.shape)
			print(agent.select_action(b.state))
			break
	test_input = torch.from_numpy(np.random.random(N_S).astype(np.float32))
	print(agent.select_action(test_input.unsqueeze(0)))
	print(test_input.unsqueeze(0).shape)
	agent.eps = 0
	print(agent.select_action(test_input.unsqueeze(0)))
	print(agent.policy_net(test_input.unsqueeze(0)))

	# Train Test
	# print(c)
	# print(a.gather(1,b))
	agent.train(1,2,2)
	# agent.load_model("../test")
	agent.save_model(0, "../test")

	# Test/Train Demo for DQN_AB
	print("Test/Train Demo for DQN_AB")
	agent = DQN_AGENT_AB(N_S, 15, [3,5], 11, None)
	for i in range(11):
		agent.mem.push(
			np.random.random(N_S).astype(np.float32), # state
			np.array([np.random.randint(3), np.random.randint(5)]).astype(np.int64), # action
			np.random.random(N_S).astype(np.float32), # next state
			np.random.random() #reward
			)
	# Inference Test
	loader = torch.utils.data.DataLoader(agent.mem,shuffle=True,batch_size=10)	
	with torch.no_grad():
		for i, b in enumerate(loader):
			print(b.state.shape)
			print(agent.select_action(b.state))
			break	
	test_input = torch.from_numpy(np.random.random(N_S).astype(np.float32))
	print(agent.select_action(test_input.unsqueeze(0)))
	print(test_input.unsqueeze(0).shape)
	agent.eps = 0
	print(agent.select_action(test_input.unsqueeze(0)))
	print(agent.policy_net(test_input.unsqueeze(0)))

	# Train Test
	agent.train(1,2,2)
	# agent.load_model("../test")
	agent.save_model(0, "../test")