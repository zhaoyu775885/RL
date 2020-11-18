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
        n_hiddens = 1024
        self.fc1 = nn.Linear(n_states, n_hiddens)
        # self.fc3 = nn.Linear(n_hiddens, n_hiddens)
        self.fc2 = nn.Linear(n_hiddens, n_actions)
        
    def forward(self, states):
        hiddens = F.relu(self.fc1(states))
        # hiddens = F.relu(self.fc3(hiddens_)) + hiddens_
        vals = self.fc2(hiddens)
        return vals


class ReplayMemory():
    def __init__(self, capacity=4096):
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
        
        self.qnet_online = ValueNetwork(self.env.dim_states, self.env.num_actions)
        self.qnet_target = ValueNetwork(self.env.dim_states, self.env.num_actions)
        self.save_model()
        self.load_model()
        
        self.loss_fn = nn.MSELoss()
        
        self.init_lr = 1e-3
        self.opt = optim.SGD(self.qnet_online.parameters(), lr=self.init_lr, 
                             momentum=0.9, weight_decay=1e-5)

        self.memory = ReplayMemory(capacity=2048)

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
        vals = self.qnet_online(torch.Tensor(s))
        return torch.argmax(vals).item()
    
    def batch(self, states):
        batch = []
        for state in states[:]:
            batch.append(state)
        return torch.Tensor(batch)
    
    def train_qnet(self, display=False):
        s, a, r, s_, status = self.memory.sample(self.batch_size)
        qvals_ = self.qnet_target(self.batch(s_))
        max_qvals_, max_indices = torch.max(qvals_, dim=1)
        max_qvals_ = max_qvals_.detach()
        done = torch.Tensor(status)
        targets = torch.Tensor(r) + self.gamma*(1-done)*max_qvals_
        
        qvals = self.qnet_online(self.batch(s))
        cur_qvals = qvals[torch.arange(self.batch_size), a]
        loss = self.metrics(targets, cur_qvals)
        if display:
            print(loss.item())
            # print(targets)
            # print(cur_qvals)
            
        self.qnet_online.zero_grad()
        loss.backward()
        self.opt.step()

    def train(self, num_eposide=10):
        for _ in range(num_eposide):
            s = self.env.init()
            cnt, done = 0, False
            while not done:
                a = self.greedy_epsilon(s)
                s_, r, done = self.env.take_action(a)
                self.memory.record([s, a, r, s_, done])
                s = s_
                if self.memory.is_full():
                    self.train_qnet() # cnt==149 and (_+1)%20==0
                    if (cnt+1) % 50 == 0:
                        self.save_model()
                        self.load_model()
                cnt += 1
                
            if (_+1) % 1 == 0:
                print('Episode ', _+1, ': ', end=' ')
                if cnt < 200:
                    print('completed with {0} steps'.format(cnt))
                else:
                    print('failed')
            
            if (_+1) % 100 == 0:
                self.epsilon *= 0.9
            
            if (_+1) % 100 == 0:
                self.test()
            

    def test(self):
        print('Policy Testing: ', end=' ')
        s = self.env.init()
        act_list = []
        cnt, done = 0, False
        while not done:
            a = self.greedy(s)
            act_list.append(a)
            s_, r, done = self.env.take_action(a)
            s = s_
            cnt += 1
        if cnt < 200:
                print('mission completed with {0} steps'.format(cnt))
        else:
            print('mission failed')
            
    def save_model(self):
        model_path = './model/model.pth'
        torch.save(self.qnet_online.state_dict(), model_path)
        
    def load_model(self):
        model_path = './model/model.pth'
        state_dict = torch.load(model_path, map_location=lambda storage, loc: storage.cuda())
        self.qnet_target.load_state_dict(state_dict)