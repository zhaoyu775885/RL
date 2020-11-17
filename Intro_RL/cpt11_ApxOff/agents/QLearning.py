# -*- coding: utf-8 -*-

import numpy as np

class QLearning():
    def __init__(self, env):
        self.env = env
        self.n_actions = self.env.num_actions
        self.q_tab = np.zeros([self.env.num_states])
        self.epsilon = 0.1
        self.alpha = 0.5
        self.gamma = 0.99
        self.debug = False
    
    def q(self, state, action):
        active_tiles = self.env.tilecoder(state, action)
        return self.q_tab[active_tiles]
    
    def greedy_epsilon(self, state):
        rand = np.random.rand()
        if rand <= self.epsilon:
            return self.env.random_action()
        return self.greedy(state)
    
    def greedy(self, state):
        g_action = 0
        g_val = self.q(state, g_action)
        for action in range(1, self.n_actions):
            q_val = self.q(state, action)
            if q_val>g_val:
                g_val, g_action = q_val, action
        return g_action

    def train(self, num_eposide=1000):
        for _ in range(num_eposide):
            print('Episode ', _+1, ': ', end=' ')
            s = self.env.init()
            cnt, done = 0, False
            act_list = []
            while not done:
                a = self.greedy_epsilon(s)
                act_list.append(a)
                s_, r, done = self.env.take_action(a)
                a_max = self.greedy(s_)
                tile_code = self.env.tilecoder(s, a)[0]
                self.q_tab[tile_code] += self.alpha*(r+self.gamma*self.q(s_, a_max)-self.q_tab[tile_code])
                s = s_
                if (cnt+1) % 50 == 0 and self.debug:
                    print(act_list)
                    act_list.clear()
                cnt += 1
            if cnt < 200:
                print('mission completed with {0} steps'.format(cnt))
            else:
                print('mission failed')
                
            if (_+1) % 100 == 0:
                self.test()
                self.epsilon *= 0.9
                self.alpha *= 0.8
                
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
                
    # def test(self):
    #     print('Policy Testing: ', end=' ')
    #     s = self.env.init()
    #     act_list = []
    #     cnt, done = 0, False
    #     while not done:
    #         a = self.greedy(s)
    #         act_list.append(a)
    #         s_, r, done = self.env.take_action(a)
    #         s = s_
    #         cnt += 1
    #     if cnt < 200:
    #         print('mission completed with {0} steps'.format(cnt))
    #     else:
    #         print('mission failed')
