#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 10 19:31:09 2019

@author: zhaoyu
"""

import numpy as np
import matplotlib.pyplot as pplot

class RandomWalk():
    def __init__(self, len_of_walk):
        self._l_bound = 0
        self._r_bound = len_of_walk+1
        self._c_pos = (self._l_bound+self._r_bound) // 2
        
    def init(self, init_pos=None):
        init_val = (self._l_bound+self._r_bound) // 2 if init_pos==None else init_pos
        self._c_pos = init_val
        
    def take_action(self, a):
        if a == 0:
            if self._c_pos == self._l_bound:
                print('error! Trying to move left at the Index 0')
            self._c_pos -= 1
        elif a == 1:
            if self._c_pos == self._r_bound:
                print('error! Trying to move right at the Index {}'.format(self._r_bound))
            self._c_pos += 1
        r = 1 if self._c_pos == self._r_bound else 0
        return r
            
    def choose_action(self):
        if self._c_pos == self._l_bound or self._c_pos == self._r_bound:
            print('error! In terminal state:', self._c_pos)
        elif self._c_pos>self._l_bound and self._c_pos<self._r_bound:
            return np.random.randint(low=0, high=2)
        return -1
        
    def game_end(self):
        if self._c_pos == self._l_bound or self._c_pos == self._r_bound:
            return True
        return False
    
    def get_c_pos(self):
        return self._c_pos
    
    def display(self):
        print(self._c_pos)
    
def TD_play(len_of_walk=5, alpha=0.1, n_episode=100):
    rw = RandomWalk(len_of_walk)
    v = np.array([0]+len_of_walk*[0.5]+[0])
    v_true = np.arange(1,len_of_walk+1) * (1/(len_of_walk+1))
    rmse = np.zeros(n_episode)
    for _ in range(n_episode):
        rw.init()
        s = rw.get_c_pos()
        while not rw.game_end():
            a = rw.choose_action()
            r = rw.take_action(a)
            s_ = rw.get_c_pos()
            v[s] += alpha * (r+v[s_]-v[s])
            s = s_
        rmse[_] = np.sqrt(1/len_of_walk*np.dot(v[1:-1]-v_true, v[1:-1]-v_true))
    return v[1:-1], rmse

def MC_play(len_of_walk=5, alpha=0.1, n_episode=100):
    '''
    no discount, gamma=1
    '''    
    rw = RandomWalk(len_of_walk)
    v = np.array([0]+len_of_walk*[0.5]+[0])
    v_true = np.arange(1,len_of_walk+1) * (1/(len_of_walk+1))
    rmse = np.zeros(n_episode)
    for _ in range(n_episode):
        rw.init()
        state_seq = []
        G = 0
        s = rw.get_c_pos()
        while not rw.game_end():
            state_seq.append(s)
            a = rw.choose_action()
            r = rw.take_action(a)
            s_ = rw.get_c_pos()
            G += r
            s = s_
        for s in state_seq:
            v[s] += alpha * (G-v[s])
#        print(state_seq, G)    
        rmse[_] = np.sqrt(1/len_of_walk*np.dot(v[1:-1]-v_true, v[1:-1]-v_true))
    return v[1:-1], rmse

if __name__ == '__main__':
    '''
    t_mat = np.mat([
            [1, 0, 0, 0, 0, 0, 0],
            [0.5, 0, 0.5, 0, 0, 0, 0],
            [0, 0.5, 0, 0.5, 0, 0, 0],
            [0, 0, 0.5, 0, 0.5, 0, 0],
            [0, 0, 0, 0.5, 0, 0.5, 0],
            [0, 0, 0, 0, 0.5, 0, 0.5],
            [0, 0, 0, 0, 0, 0, 1]
            ])
    t_mat_inf = t_mat**1000
    '''
    n_run = 100
    len_of_walk = 5
    n_episode = 100
    TD_alpha_list = [0.15, 0.1, 0.05]
    MC_alpha_list = [0.01, 0.02, 0.03, 0.04, 0.05]
    vf, rmse = MC_play(len_of_walk=len_of_walk, alpha=0.1, n_episode=n_episode)
    
    for alpha in TD_alpha_list:
        mean_rmse = np.zeros(n_episode)
        for _ in range(n_run):
            vf, rmse = TD_play(len_of_walk=len_of_walk, alpha=alpha, n_episode=n_episode)
            mean_rmse += rmse
        mean_rmse /= n_run
        pplot.plot(mean_rmse)
        
    for alpha in MC_alpha_list:
        mean_rmse = np.zeros(n_episode)
        for _ in range(n_run):
            vf, rmse = MC_play(len_of_walk=len_of_walk, alpha=alpha, n_episode=n_episode)
            mean_rmse += rmse
        mean_rmse /= n_run
        pplot.plot(mean_rmse)