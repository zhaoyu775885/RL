#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 19:43:58 2019

@author: zhaoyu
"""

import gym
#env = gym.make('CartPole-v0')
env = gym.make('MountainCar-v0')
for i_episode in range(1):
    observation = env.reset()
    for step in range(100):
        env.render()
        # action = env.action_space.sample()
        action = 2
        observation, reward, done, info = env.step(action)
        print(action, observation, reward)
        if done:
            print("Episode finished after {} timesteps".format(step+1))
            break
        
env.close()