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
        self._l_bnd = 0
        self._r_bnd = 2
        self.end = self._r_bnd+self.MOVE_RIGHT
        self.init()
        
    def init(self):
        self.pos = self._l_bnd
        return self.pos
        
    def take_action(self, action):
        if self.pos == self._l_bnd:
            self.pos = self._l_bnd if action==self.MOVE_LEFT else self._l_bnd+self.MOVE_RIGHT
        elif self.pos == 1:
            self.pos -= action
        elif self.pos == self._r_bnd:
            self.pos += action
        return -1
    
    def choose_action(self, p):
        return self.MOVE_RIGHT if np.random.rand()<=p else self.MOVE_LEFT
    
    def action(self, p):
        a = self.choose_action(p)
        r = self.take_action(a)
        s = self.get_state()
        return a, r, s
    
    def game_end(self):
        return True if self.pos == self.end else False
    
    def get_state(self):
        return self.pos
    
    def __str__(self):
        return '{0}'.format(self.pos)
    
    
if __name__ == '__main__':
    env = GridWorld()
    print('start at pos: ', env)
    
    params = np.random.rand()
    print(params)
    for _ in range(1000):
        if env.game_end():
            break
        a, s, r = env.action([params])
        print('step:', _, 'arrive at pos: ', env)
        
        