#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 15:32:08 2020

@author: zhaoyu
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
import numpy as np


class ValueNetwork(nn.Module):
    def __init__(self, n_states, n_actions):
        super(ValueNetwork, self).__init__()
        n_hiddens = 500
        self.fc1 = nn.Linear(n_states, n_hiddens)
        self.fc2 = nn.Linear(n_hiddens, n_actions)
        
    def forward(self, states):
        hiddens = F.relu(self.fc1(states))
        vals = self.fc2(hiddens)
        return vals


class ReplayMemory():
    def __init__(self, capacity=256):
        self.memory = deque()
        self.capacity = capacity
        
    def record(self, play_info):
        if self.size() == self.capacity:
            self.memory.popleft()
        self.memory.append(play_info)
        
    def size(self):
        return len(self.memory)
    
    def is_full(self):
        return self.capacity == self.size()
    
    def sample(self, batch_size=16):
        assert batch_size<self.capacity
        assert self.is_full()
        samples = np.random.choice(self.memory, batch_size, replace=False)
        bat_s, bat_a, bat_r, bat_s_ = [], [], [], []
        for s, a, r, s_ in samples:
            bat_s.append(s)
            bat_a.append(a)
            bat_r.append(r)
            bat_s_.append(s_)
        return bat_s, bat_a, bat_r, bat_s_


class DeepQNetwork():
    def __init__(self, env):
        self.env = env
        self.n_actions = self.env.num_actions
        
        self.qnet = ValueNetwork(self.env.num_states, self.n_actions)
        self.loss_fn = nn.MSELoss()
        
        self.init_lr = 1e-3
        self.opt = optim.SGD(self.qnet.parameters(), lr=self.init_lr, 
                             momentum=0.9, weight_decay=1e-5)
        
        self.memory = ReplayMemory(capacity=256)
        
        self.epsilon = 0.05
        self.gamma = 0.99
        self.debug = False
    
    def metrics(self, targets, qvals):
        return self.loss_fn(targets, qvals)

    def greedy_epsilon(self, state):
        rand = np.random.rand()
        if rand <= self.epsilon:
            return self.env.random_action()
        return self.greedy(state)
    
    def greedy(self, state):
        vals = self.qnet(state)
        return torch.max(vals)

    def train(self, num_eposide=1000):
        for _ in range(num_eposide):
            print('Episode ', _+1, ': ', end=' ')
            s = self.env.init()
            
            # cnt = 0
            # act_list = []
            while True:
                # cnt += 1
                a = self.greedy_epsilon(s)
                # act_list.append(a)
                s_, r, done = self.env.take_action(a)
                self.memory.record([s, a, r, s_])
                
                if done:
                    
                
                if self.memory.is_full():
                    # train Q-network
                    s, a, r, s_ = self.memory.sample(16)
                    targets = self.greedy(hotcode(s_))*self.gamma + torch.Tensor(r)
                    qvals = self.qnet(hotcode(s))
                    loss = self.metrics(targets, qvals[a])
                    self.qnet.zero_grad()
                    loss.backward()
                    self.opt.step()
                s = s_
                
                # if done:
                #     if cnt < 200:
                #         print('mission completed with {0} steps'.format(cnt))
                #     else:
                #         print('mission failed')
                #     break
                
            if (_+1) % 100 == 0:
                self.test()
                self.epsilon *= 0.9
                self.alpha *= 0.9
            
            