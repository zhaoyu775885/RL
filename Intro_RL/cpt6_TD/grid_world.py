# -*- coding: utf-8 -*-

import numpy as np
import time

# class grid_world(ABC)
# abstract base class

class GridWorld():
    '''
    define the environment of the grid world
    involves: 
        take_action: return the reward and next state
        effect of other causes, like wind
        judge whether the goal is reached
        display the current situation of the environment
    do not involve:
        q_table, which should be part of the policy and learning algorithm
    '''    
    def __init__(self, acts=[0, 1, 2, 3], panel=[7, 10], start=None, end=None):
        self._n_row, self._n_col = panel
        self._n_state = self._n_row * self._n_col
        self._acts = acts
        self._n_act = len(self._acts)
        if start != None:
            self._s_row, self._s_col = start
        if end != None:
            self._g_row, self._g_col = end

    def init(self):
        self._c_row, self._c_col = self._s_row, self._s_col

    def the_end(self):
        if [self._c_row, self._c_col]== [self._g_row, self._g_col]:
            return True
        return False

    def set_state(self, s_row, s_col):
        self._c_row, self._c_col = s_row, s_col
        
    def get_state(self):
        return self._c_row*self._n_col+self._c_col

    def get_num_state(self):
        return self._n_state
    
    def get_num_action(self):
        return self._n_act
    
    def get_actions(self):
        return self._acts

class WindyGridWorld(GridWorld):
    def __init__(self, acts, panel=[7, 10], start=None, end=None, 
                 wind=None, stochastic_wind=False):
        super(WindyGridWorld, self).__init__(acts, panel, start, end)
        self._wind = wind
        if len(self._wind) != self._n_col:
            raise Exception('Invalid wind definition', self._wind)
        self.stochastic_wind = stochastic_wind
    
    def wind_effect(self):
        wind = self._wind[self._c_col]
        if self.stochastic_wind and wind!=0:
            stochastic_error = np.random.choice([-1, 0, 1])
            wind += stochastic_error
        for _ in range(wind):
            self._c_row = self._c_row-(0 if self._c_row==0 else 1)        
    
    def take_action(self, a):
        self.wind_effect()
        if a == 0:
            self._c_row = self._c_row-(0 if self._c_row==0 else 1)
        if a == 1:
            self._c_col = self._c_col+(0 if self._c_col==self._n_col-1 else 1)
        if a == 2:
            self._c_row = self._c_row+(0 if self._c_row==self._n_row-1 else 1)
        if a == 3:
            self._c_col = self._c_col-(0 if self._c_col==0 else 1)
        if a == 4:
            self._c_row = self._c_row-(0 if self._c_row==0 else 1)
            self._c_col = self._c_col+(0 if self._c_col==self._n_col-1 else 1)
        if a == 5:
            self._c_col = self._c_col+(0 if self._c_col==self._n_col-1 else 1)
            self._c_row = self._c_row+(0 if self._c_row==self._n_row-1 else 1)
        if a == 6:
            self._c_row = self._c_row+(0 if self._c_row==self._n_row-1 else 1)
            self._c_col = self._c_col-(0 if self._c_col==0 else 1)
        if a == 7:
            self._c_col = self._c_col-(0 if self._c_col==0 else 1)
            self._c_row = self._c_row-(0 if self._c_row==0 else 1)
        if a == 8:
            return 0
        return -1
    
class CliffGridWorld(GridWorld):
    def __init__(self, acts, panel, start=None, end=None):
        super(CliffGridWorld, self).__init__(acts, panel, start, end)
        
    def take_action(self, a):
        def on_cliff():
            if self._c_row == self._g_row and self._c_col>self._s_col and self._c_col<self._g_col:
                return True
            return False
        if a == 0:
            self._c_row = self._c_row-(0 if self._c_row==0 else 1)
        if a == 1:
            self._c_col = self._c_col+(0 if self._c_col==self._n_col-1 else 1)
        if a == 2:
            self._c_row = self._c_row+(0 if self._c_row==self._n_row-1 else 1)
        if a == 3:
            self._c_col = self._c_col-(0 if self._c_col==0 else 1)
        
        if on_cliff():
            self.init()
            return -100
        return -1

class TD():
    def __init__(self, env, epsilon=0.1, alpha=0.5):
        self.env = env
        self.epsilon = epsilon
        self.alpha = alpha
        self._q = self.init_qtable()
    
    def init_qtable(self):
        n_state, n_act = None, None
        if self.env != None:
            n_state = self.env.get_num_state()
            n_act = self.env.get_num_action()
            return np.zeros([n_state, n_act])
        else:
            raise Exception('Get env parameters [n_state, n_act] error!')
        return None
    
    def choose_action(self):
        c_state_idx = self.env.get_state()
        rd = np.random.uniform(0, 1)
        if rd <= self.epsilon:
            a = np.random.choice(self.env.get_actions())
        else:
            max_val = np.max(self._q[c_state_idx, :])
            max_val_indices = []
            for idx, val in enumerate(self._q[c_state_idx, :]):
                if val == max_val:
                    max_val_indices.append(idx)
            a = np.random.choice(max_val_indices)
        return a
    
    def Sarsa_update(self, s, a, r, s_, a_):
        self._q[s, a] += self.alpha*(r+self._q[s_, a_]-self._q[s, a])
        
    def Qlearning_update(self, s, a, r, s_):
        self._q[s, a] += self.alpha*(r+np.max(self._q[s_, :])-self._q[s, a])

    def ESarsa_update(self, s, a, r, s_):
        v_s_ = np.max(self._q[s_, :])*(1-self.epsilon) + \
            np.sum(self._q[s_, :])*self.epsilon/self.env.get_num_action()
        self._q[s, a] += self.alpha*(r+v_s_-self._q[s, a])        
    
    def greedy(self):
        n_step = 0
        self.epsilon = 0
        self.env.init()
        s = self.env.get_state()
        a = self.choose_action()
        while not self.env.the_end():
            r = self.env.take_action(a)
            print(s, a, ':', self._q[s, :])
            s_ = self.env.get_state()
            a_ = self.choose_action()
            s, a = s_, a_
            n_step += 1 if a != 8 else 0
        print('steps: ', n_step)
    
    def Sarsa(self, n_episode=100):
        for _ in range(n_episode):
            n_step = 0
            self.env.init()
            s = self.env.get_state()
            a = self.choose_action()
            g = 0
            while not self.env.the_end():
                r = self.env.take_action(a)
                s_ = self.env.get_state()
                a_ = self.choose_action()
                self.Sarsa_update(s, a, r, s_, a_)
                if (_+1) % 100 == 0:
                    print(s_,  end=' ')
                s, a = s_, a_
                g += r
                n_step += 1
            if (_+1) % 100 == 0:
                print('\n=====', _+1, ':', n_step, ', G: ', g, '=======\n')
                
    def Qlearning(self, n_episode=100):
        for _ in range(n_episode):
            n_step = 0
            self.env.init()
            s = self.env.get_state()
            g = 0
            while not self.env.the_end():
                a = self.choose_action()
                r = self.env.take_action(a)
                s_ = self.env.get_state()
                self.Qlearning_update(s, a, r, s_)
                if (_+1) % 100 == 0:
                    print(s_,  end=' ')                
                s = s_
                g += r
                n_step += 1
            if (_+1) % 100 == 0:
                print('\n=====', _+1, ':', n_step, ', G: ', g, '=======\n')
                
    def ESarsa(self, n_episode=100):
        for _ in range(n_episode):
            n_step = 0
            self.env.init()
            s = self.env.get_state()
            g = 0
            while not self.env.the_end():
                a = self.choose_action()
                r = self.env.take_action(a)
                s_ = self.env.get_state()
                self.ESarsa_update(s, a, r, s_)
                if (_+1) % 100 == 0:
                    print(s_,  end=' ')         
                s = s_
                g += r
                n_step += 1
            if (_+1) % 100 == 0:
                print('\n=====', _+1, ':', n_step, ', G: ', g, '=======\n')
                
    def show_qtable(self):
        print(self._q)
        
        
def queen_move():
    # define windy_grid_world
    wind = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]
    env = WindyGridWorld(acts=[0,1,2,3], panel=[7,10], start=[3,0], end=[3,7], wind=wind)
    
    # define TD learner
    lrner = TD(env)
    
    # use Sarsa control
    lrner.Sarsa(2000)
    lrner.greedy()
    
    
def king_move():
    # define windy_grid_world
    wind = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]
    env = WindyGridWorld(acts=[0,1,2,3,4,5,6,7], panel=[7,10], start=[3,0], end=[3,7], wind=wind)
    
    # define TD learner
    lrner = TD(env)
    
    # use Sarsa control
    lrner.Sarsa(2000)
    lrner.greedy()
    
    
def cliff_walking():
    # define cliff_grid_world
    env = CliffGridWorld(acts=[0, 1, 2, 3], panel=[4,12], start=[3,0], end=[3,11])
    
    # define TD learner
    lrner = TD(env)
    
    # use TD control [Sarsa, Qlearning, Expected-Sarsa]
#    lrner.Sarsa(1000)    
#    lrner.Qlearning(1000)
    lrner.ESarsa(1000)
    lrner.greedy()
    
    
if __name__ == '__main__':
#    queen_move()
#    king_move()
    cliff_walking()