#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 11:09:45 2020

@author: zhaoyu
"""

import numpy as np

class Reinforce():
    def __init__(self, env, gamma=1):
        self.env = env
        self.gamma = gamma
        self.params = np.random.rand(1)
        
    def train(self, init_lr=1e-3):
        def step(s_list, a_list, r_list):
            a, s, r = self.env.action(params=self.params)
            s_list.append(s)
            a_list.append(a)
            r_list.append(r)
            
        for _ in range(100):
            s_list = []
            a_list = []
            r_list = []
            
            step(s_list, a_list, r_list)
            while not self.env.the_end():
                step(s_list, a_list, r_list)
                
            gt = 1
            for t in range(len(s_list)):
                self.params += lr*
            
            
            