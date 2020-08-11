#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 10:57:54 2020

@author: zhaoyu
"""

from SimGridWorld import GridWorld
from PG import Reinforce
import matplotlib.pyplot as plt

if __name__ == '__main__':
    print('Policy Graident Demo', '\n=======================')
    
    env = GridWorld()
    print('Init State: \n', env)
    pg = Reinforce(env, gamma=1)
    
    n_rollout = 1000
    g0 = pg.train(rollout=n_rollout, init_lr=2**-13)
    plt.plot(g0)