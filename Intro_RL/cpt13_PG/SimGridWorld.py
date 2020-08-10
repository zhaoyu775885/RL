#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 11:10:02 2020

@author: zhaoyu
"""

import numpy as np

class GridWorld():
    MOVE_RIGHT = 1
    MOVE_LEFT = -1
    def __init__(self):
        self.left = 0
        self.right = 2
        self.end = self.right+self.MOVE_RIGHT
        self.pos = self.left
        
    def take_action(self, action):
        if self.pos == self.left:
            self.pos = self.left if action==self.MOVE_LEFT else self.left+self.MOVE_RIGHT
        elif self.pos == 1:
            self.pos -= action
        elif self.pos == self.right:
            self.pos += action
        return self.pos, -1
    
    def choose_action(self, params):
        return self.MOVE_RIGHT if np.random.rand()<=params[0] else self.MOVE_LEFT
    
    def action(self, params):
        a = self.choose_action(params)
        s, r = self.take_action(a)
        return a, s, r
    
    def the_end(self):
        return True if self.pos == self.end else False
    
    def __str__(self):
        return '{0}'.format(self.pos)
    
    
if __name__ == '__main__':
    env = GridWorld()
    print('start at pos: ', env)
    
    for _ in range(100):
        if env.the_end():
            break
        a, s, r = env.action([0.5])
        print('step:', _, 'arrive at pos: ', env)
        
        