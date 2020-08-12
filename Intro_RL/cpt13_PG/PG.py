#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 11:09:45 2020

@author: zhaoyu
"""

import numpy as np
import matplotlib.pyplot as plt

def sigmoid(theta):
    return 1/(1+np.exp(-theta))

class Reinforce():
    def __init__(self, env, gamma=1):
        self.env = env
        self.gamma = gamma
        self.params = None
        self.init_params()
        
    def init_params(self):
        # self.params = np.random.randn(1)*4
        self.params = np.array([3], dtype=np.float32)
        
    def step(self, s_list, a_list, r_list):
            a, r, s = self.env.action(p=sigmoid(self.params[0]))
            s_list.append(s)
            a_list.append(a)
            r_list.append(r)
            
    def play(self, n_run=100):
        rollout_return = np.zeros(n_run)
        for _ in range(n_run):
            self.env.init()
            s_list = []
            a_list = []
            r_list = []
            while not self.env.game_end():
                self.step(s_list, a_list, r_list)
                
            g_list = np.zeros(len(r_list))
            g_list[-1] = r_list[-1]
            for i in range(len(r_list)-1):
                r_idx = len(r_list)-i-2
                g_list[r_idx] = r_list[r_idx]+self.gamma*g_list[r_idx+1]
                
            rollout_return[_] = g_list[0]
        return rollout_return.mean()
            
    def train(self, rollout=1000, init_lr=1e-4):
        rollout_return = []
        
        lr = init_lr
        for _ in range(rollout):
            g0 = self.play(n_run=1000)
            rollout_return.append(g0)
            
            self.env.init()
            s_list = []
            a_list = []
            r_list = []
            while not self.env.game_end():
                self.step(s_list, a_list, r_list)
                
            g_list = np.zeros(len(r_list))
            g_list[-1] = r_list[-1]
            for i in range(len(r_list)-1):
                r_idx = len(r_list)-i-2
                g_list[r_idx] = r_list[r_idx]+self.gamma*g_list[r_idx+1]
                
            # print(s_list, a_list, r_list, g_list)
            
            for t in range(len(s_list)):
                p = sigmoid(self.params[0])
                grad_pi = 1-p if a_list[t]==self.env.MOVE_RIGHT else -p
                # self.params += lr*(g_list[t])*grad_pi
                self.params += lr*(g_list[t]-g0)*grad_pi
                # todo: with MC estimation

            print(_, g_list[0], self.params, sigmoid(self.params[0]))
            
        # plt.plot(rollout_return)
        return rollout_return