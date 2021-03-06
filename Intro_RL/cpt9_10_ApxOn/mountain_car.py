import gym
from tiles3 import tiles, IHT
import numpy as np

def run_demo():
    env = gym.make('MountainCar-v0') #导入MountainCar-v0环境
    env.reset() #初始化环境
    for _ in range(100): #循环100次
        env.render() #绘图
        a = env.action_space.sample() #进行一个动作
        info = env.step(a) # take a random action
    env.close() #关闭
    
class MountainCar():
    def __init__(self, 
                 num_tilings=8, 
                 num_tiles_one_dim=8,
                 num_states=4096):
        self.env = gym.make('MountainCar-v0')
        self.num_states = num_states
        self.iht = IHT(self.num_states)
        self.num_tilings = num_tilings
        self.num_tiles_one_dim = num_tiles_one_dim
        self.obs_ranges = [h-l for h,l in zip(self.env.observation_space.high,
                                              self.env.observation_space.low)]
        self.num_actions = self.env.action_space.n
        self.state = self.init()
        
    def tilecoder(self, obs, act=0):
        return tiles(self.iht, self.num_tilings, 
                     [self.num_tiles_one_dim* obs_dim / range_dim
                      for obs_dim, range_dim in zip(obs, self.obs_ranges)],
                     [act])
    
    def init(self):
        obs = self.env.reset()
        # self.env.render()
        return obs
    
    def get_state(self):
        return self.state
        
    def random_action(self):
        return self.env.action_space.sample()
        
    def take_action(self, a):
        s, r, status, info = self.env.step(a)
        # self.env.render()
        self.state = s
        return s, r, status

    def close(self):
        if self.env is not None:
            self.env.close()
            self.env = None
        
    def __del__(self):
        print('running destructor')
        self.close()


class SemiGradSarsa():
    def __init__(self, env):
        self.env = env
        self.n_actions = self.env.num_actions
        self.weight = np.zeros([self.env.num_states])
        self.epsilon = 0.1
        self.alpha = 0.5/8
        self.gamma = 0.99
        self.debug = False
    
    def q(self, state, action):
        active_tiles = self.env.tilecoder(state, action)
        return np.sum(self.weight[active_tiles])
    
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
    
    def train(self):
        num_eposide = 100
        for _ in range(num_eposide):
            print('Episode ', _+1, ': ', end=' ')
            s = self.env.init()
            a = self.greedy_epsilon(s)
            act_list = [a]
            cnt = 0
            while True:
                cnt += 1
                tile_code = self.env.tilecoder(s, a)
                s_, r, done = self.env.take_action(a)
                if done:
                    if cnt < 200:
                        print('completed with {0} steps'.format(cnt))
                    else:
                        print('failed')
                    self.weight[tile_code] += self.alpha*(r-self.q(s, a))
                    break
                a_ = self.greedy_epsilon(s_)
                act_list.append(a_)
                self.weight[tile_code] += self.alpha*(r+self.gamma*self.q(s_, a_)-self.q(s, a))
                s, a = s_, a_
                if cnt % 50 == 0 and self.debug:
                    print(act_list)
                    act_list.clear()
            if (_+1) % 5 == 0:
                self.test()
                self.epsilon *= 0.5
                
    def test(self):
        print('Policy Testing: ', end=' ')
        s = self.env.init()
        act_list = []
        cnt = 0
        while True:
            cnt += 1
            a = self.greedy(s)
            act_list.append(a)
            s_, r, done = self.env.take_action(a)
            s = s_
            if done:
                if cnt < 200:
                    print('mission completed with {0} steps'.format(cnt))
                else:
                    print('mission failed')
                break
            

class QLearning():
    def __init__(self, env):
        self.env = env
        self.n_actions = self.env.num_actions
        self.q_tab = np.zeros([self.env.num_states])
        self.epsilon = 0.05
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
            
            cnt = 0
            act_list = []
            while True:
                cnt += 1
                a = self.greedy_epsilon(s)
                act_list.append(a)
                tile_code = self.env.tilecoder(s, a)[0]
                s_, r, done = self.env.take_action(a)
                a_max = self.greedy(s_)
                if (cnt+1) % 50 == 0 and self.debug:
                    print(act_list)
                    act_list.clear()
                self.q_tab[tile_code] += self.alpha*(r+self.gamma*self.q(s_, a_max)-self.q_tab[tile_code])
                s = s_
                
                if done:
                    if cnt < 200:
                        print('mission completed with {0} steps'.format(cnt))
                    else:
                        print('mission failed')
                    break
                
            if (_+1) % 100 == 0:
                self.test()
                self.epsilon *= 0.9
                self.alpha *= 0.9
                
    def test(self):
        print('Policy Testing: ', end=' ')
        s = self.env.init()
        act_list = []
        cnt = 0
        while True:
            cnt += 1
            a = self.greedy(s)
            act_list.append(a)
            s_, r, done = self.env.take_action(a)
            s = s_
            if done:
                if cnt < 200:
                    print('mission completed with {0} steps'.format(cnt))
                else:
                    print('mission failed')
                break
            

if __name__ == '__main__':
    # Algo 1: Semi-Gradient Sarsa
    env = MountainCar(num_tilings=8, num_tiles_one_dim=8, num_states=4096)
    leaner = SemiGradSarsa(env)
    leaner.train()
    env.close()
    
    # # Algo 2: 
    # env = MountainCar(num_tilings=1, num_tiles_one_dim=32, num_states=10000)
    # leaner = QLearning(env)
    # leaner.train(num_eposide=10000)
    # env.close()
