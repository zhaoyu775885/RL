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
import random


class ValueNetwork(nn.Module):
    def __init__(self, n_states, n_actions):
        super(ValueNetwork, self).__init__()
        n_hiddens = 100
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
        samples = random.sample(self.memory, batch_size)
        bat_s, bat_a, bat_r, bat_s_, bat_done = [], [], [], [], []
        for s, a, r, s_, done in samples:
            bat_s.append(s)
            bat_a.append(a)
            bat_r.append(r)
            bat_s_.append(s_)
            bat_done.append(done)
        return bat_s, bat_a, bat_r, bat_s_, bat_done


class DeepQLearning():
    def __init__(self, env):
        self.env = env
        self.n_actions = self.env.num_actions
        
        self.qnet = ValueNetwork(self.env.num_states, self.n_actions)
        self.loss_fn = nn.MSELoss()
        
        self.init_lr = 1e-3
        self.opt = optim.SGD(self.qnet.parameters(), lr=self.init_lr, 
                             momentum=0.9, weight_decay=1e-5)
        
        self.memory = ReplayMemory(capacity=4096)
        
        self.batch_size = 32
        self.epsilon = 0.1
        self.gamma = 0.99
        self.debug = False
    
    def metrics(self, targets, qvals):
        return self.loss_fn(qvals, targets)

    def greedy_epsilon(self, state):
        rand = np.random.rand()
        if rand <= self.epsilon:
            return self.env.random_action()
        return self.greedy(state)
    
    def greedy(self, s):
        hotcode = self.env.hotcoder(s)
        vals = self.qnet(torch.Tensor(hotcode))
        return torch.argmax(vals).item()
    
    def batch_hotcode(self, states):
        hotcodes = []
        for state in states[:]:
            hotcodes.append(self.env.hotcoder(state))
        return torch.Tensor(hotcodes)
    
    def train_qnetwork(self):
        s, a, r, s_, status = self.memory.sample(self.batch_size)
        qvals = self.qnet(self.batch_hotcode(s))
        qvals_ = self.qnet(self.batch_hotcode(s_))
        done = torch.Tensor(status)
        max_qvals_, max_indices = torch.max(qvals_, dim=1)
        max_qvals_ = max_qvals_.detach()
        targets = torch.Tensor(r) + self.gamma*(1-done)*max_qvals_
        cur_qvals = qvals[torch.arange(self.batch_size), a]
        # print(targets, cur_qvals)
        loss = self.metrics(targets, cur_qvals)
        self.qnet.zero_grad()
        loss.backward()
        self.opt.step()

    def train(self, num_eposide=10):
        for _ in range(num_eposide):
            print('Episode ', _+1, ': ', end=' ')
            s = self.env.init()
            cnt = 0
            while True:
                cnt += 1
                a = self.greedy_epsilon(s)
                s_, r, done = self.env.take_action(a)
                self.memory.record([s, a, r, s_, done])
                s = s_
                if self.memory.is_full():
                    self.train_qnetwork()
                if done:
                    if cnt < 200:
                        print('completed with {0} steps'.format(cnt))
                    else:
                        print('failed')
                    break
            if (_+1) % 100 == 0:
                self.test()
                self.epsilon *= 0.9
                
    def test(self):
        print('Policy Testing: ', end=' ')
        s = self.env.init()
        cnt = 0
        while True:
            cnt += 1
            a = self.greedy(s)
            s_, r, done = self.env.take_action(a)
            s = s_
            if done:
                if cnt < 200:
                    print('mission completed with {0} steps'.format(cnt))
                else:
                    print('mission failed')
                break        
