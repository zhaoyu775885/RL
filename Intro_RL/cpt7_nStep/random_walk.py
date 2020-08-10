#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 4 10:31:09 2019

@author: zhaoyu
"""

import numpy as np
import matplotlib.pyplot as pplot
import sys
sys.path.append("..")
from cpt6_TD.random_walk import RandomWalk

def n_step_return(queue, gamma=1):
    coefs = np.power(gamma, np.arange(len(queue)))
    return np.dot(queue, coefs)
    
def TD_play(len_of_walk=5, alpha=0.1, n_episode=100, gamma=1, n_step=1):
    rw = RandomWalk(len_of_walk)
    v = np.array([0]+len_of_walk*[0.5]+[0])
    v_true = np.arange(1,len_of_walk+1) * (1/(len_of_walk+1))
    rmse = np.zeros(n_episode)
    for _ in range(n_episode):
        rw.init()
        s_queue = [] # maintain one queue recording n_step states
        r_queue = [] # maintain one queue recording n_step rewards
        t = 0
        s = rw.get_c_pos()
        while not rw.game_end():
            s_queue.append(s)
            a = rw.choose_action()
            r = rw.take_action(a)
            s_ = rw.get_c_pos()
            r_queue.append(r)
            t += 1
            if t >= n_step:
                TD_target = n_step_return(r_queue, gamma)+np.power(gamma, n_step)*v[s_]
                s_n_step_prev = s_queue[0]
                v[s_n_step_prev] += alpha * (TD_target-v[s_n_step_prev])
                s_queue.pop(0)
                r_queue.pop(0)
            s = s_
        # time index should exactly correspond with each other: [state, reward]
        while len(r_queue)>0:
            TD_target = n_step_return(r_queue, gamma)
            s_n_step_prev = s_queue[0]
            v[s_n_step_prev] += alpha * (TD_target-v[s_n_step_prev])
            s_queue.pop(0)
            r_queue.pop(0)
        rmse[_] = np.sqrt(1/len_of_walk*np.dot(v[1:-1]-v_true, v[1:-1]-v_true))
    return v[1:-1], rmse

if __name__ == '__main__':
    '''
    n-step TD method bridges the TD (one-step) and the Monte Carlo methods smoothly
    '''
    n_run = 100
    len_of_walk = 5
    n_episode = 100
    TD_alpha_list = [0.15, 0.1, 0.05]
    MC_alpha_list = [0.01, 0.02, 0.03, 0.04, 0.05]
    
    for alpha in TD_alpha_list:
        mean_rmse = np.zeros(n_episode)
        for _ in range(n_run):
            vf, rmse = TD_play(len_of_walk=len_of_walk, alpha=alpha, n_episode=n_episode, n_step=1)
            mean_rmse += rmse
        mean_rmse /= n_run
        pplot.plot(mean_rmse)
        
    for alpha in MC_alpha_list:
        mean_rmse = np.zeros(n_episode)
        for _ in range(n_run):
            vf, rmse = TD_play(len_of_walk=len_of_walk, alpha=alpha, n_episode=n_episode, n_step=1000)
            mean_rmse += rmse
        mean_rmse /= n_run
        pplot.plot(mean_rmse)        