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
    def __init__(self, num_tilings=8):
        self.env = gym.make('MountainCar-v0').env
        self.num_states = 4096
        self.iht = IHT(self.num_states)
        self.num_tilings = num_tilings
        self.num_tiles_one_dim = 8
        self.num_actions = self.env.action_space.n
        self.obs_ranges = [h-l for h,l in zip(self.env.observation_space.high,
                                              self.env.observation_space.low)]
        self.state = self.init()
        
    def tilecoder(self, obs, act=0):
        return tiles(self.iht, self.num_tilings, 
                     [self.num_tiles_one_dim* obs_dim / range_dim
                      for obs_dim, range_dim in zip(obs, self.obs_ranges)],
                     [act])
    
    def init(self):
        obs = self.env.reset()
        self.env.render()
        return obs
    
    def get_state(self):
        return self.state
        
    def random_action(self):
        return self.env.action_space.sample()
        
    def take_action(self, a):
        obs, r, status, info = self.env.step(a)
        self.env.render()
        self.state = obs
        return obs, r, status

    def close(self):
        if self.env:
            self.env.close()
        self.env = None
                

class SemiGradSarsa():
    def __init__(self, env):
        self.env = env
        self.n_actions = self.env.num_actions
        self.weight = np.zeros([self.env.num_states])
        self.epsilon = 0.1
        self.alpha = 0.1
        self.gamma = 0.99
    
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
        num_eposide = 20
        for _ in range(num_eposide):
            s = self.env.init()
            a = self.greedy_epsilon(s)
            cnt = 0
            while True:
                cnt += 1
                s_, r, done = self.env.take_action(a)
                tile_code = self.env.tilecoder(s, a)
                if done:
                    print('Game Over', cnt)
                    self.weight[tile_code] += self.alpha*(r-self.q(s, a))
                    break
                a_ = self.greedy_epsilon(s_)
                # if (cnt+1) % 100 == 0:
                #     print(self.weight)
                self.weight[tile_code] += self.alpha*(r+self.gamma*self.q(s_, a_)-self.q(s, a))
                s, a = s_, a_
            print(_+1, '-th trial')
            # print(self.weight)

    
if __name__ == '__main__':
    env = MountainCar()
    leaner = SemiGradSarsa(env)
    leaner.train()
    env.close()
