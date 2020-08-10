# -*- coding: utf-8 -*-

import numpy as np
import sys
sys.path.append("..")
from cpt6_TD.random_walk import RandomWalk
from cpt7_nStep.random_walk import n_step_return

class RandomWalk1000(RandomWalk):
    def __init__(self):
        super(RandomWalk1000, self).__init__(1000)

    def take_action(self, a):
        self._c_pos += a
        if self._c_pos <= self._l_bound:
            self._c_pos = self._l_bound
        elif self._c_pos >= self._r_bound:
            self._c_pos = self._r_bound
        r = 1 if self._c_pos == self._r_bound else (-1 if self._c_pos==self._l_bound else 0)
        return r
    
    def choose_action(self):
        if self._c_pos == self._l_bound or self._c_pos == self._r_bound:
            print('error! In terminal state:', self._c_pos)
        elif self._c_pos>self._l_bound and self._c_pos<self._r_bound:
            sign = np.random.choice([-1, 1])
            act_val = np.random.randint(low=0, high=101)
            action = sign*act_val
        return action
    
def Semi_gradient_TD(alpha=0.1, n_episode=1000, gamma=1, n_step=1):
    rw = RandomWalk1000()
    v_hat = np.zeros(12)
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
                TD_target = n_step_return(r_queue, gamma)+np.power(gamma, n_step)*v_hat[(s_-1)//100+1]
                s_n_step_prev = s_queue[0]
                v_hat[(s_n_step_prev-1)//100+1] += alpha * (TD_target-v_hat[(s_n_step_prev-1)//100+1])
                s_queue.pop(0)
                r_queue.pop(0)
            s = s_
        while len(r_queue)>0:
            TD_target = n_step_return(r_queue, gamma)
            s_n_step_prev = s_queue[0]
            v_hat[(s_n_step_prev-1)//100+1] += alpha * (TD_target-v_hat[(s_n_step_prev-1)//100+1])
            s_queue.pop(0)
            r_queue.pop(0)            
    return v_hat[1:-1]
        
if __name__ == '__main__':
    v_hat = Semi_gradient_TD(n_step=4)