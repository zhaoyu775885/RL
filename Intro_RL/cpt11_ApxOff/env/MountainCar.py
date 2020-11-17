import gym
from utils.tiles3 import tiles, IHT
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
                 render=True,
                 tilecoding=False,
                 num_tilings=8,
                 num_tiles_one_dim=8):
        
        self.env = gym.make('MountainCar-v0')
        self.render = render
        self.tilecoding = tilecoding
        
        self.state = self.init()
        self.num_actions = self.env.action_space.n
        self.dim_states = self.env.observation_space.shape[0]
        
        if self.tilecoding:
            self.num_tilings = num_tilings
            self.num_tiles_one_dim = num_tiles_one_dim
            num_states = self.num_tilings*self.num_actions*2
            for _ in range(self.dim_states):
                num_states *= self.num_tiles_one_dim
            self.num_states = num_states
            self.iht = IHT(self.num_states)
            self.obs_ranges = [h-l for h,l in zip(self.env.observation_space.high,
                                                  self.env.observation_space.low)]
            
        
        print('|*********** INFO ***********|')
        print('Env=', 'MountainCar-v0', 
              'dim_state=', self.dim_states, 
              'num_actions=', self.num_actions)
        print('|----------------------------|\n')
        
        
    def init(self):
        obs = self.env.reset()
        if self.render:
            self.env.render()
        return obs

    def get_state(self):
        return self.state

    def take_action(self, a):
        s, r, status, info = self.env.step(a)
        if self.render:
            self.env.render()
        self.state = s
        return s, r, status
           
    def tilecoder(self, obs, act=None):
        if act is None:
            return tiles(self.iht, self.num_tilings,
                         [self.num_tiles_one_dim* obs_dim / range_dim
                          for obs_dim, range_dim in zip(obs, self.obs_ranges)])
        else:
            return tiles(self.iht, self.num_tilings, 
                         [self.num_tiles_one_dim* obs_dim / range_dim
                          for obs_dim, range_dim in zip(obs, self.obs_ranges)],
                         [act])
        
    # def hotcoder(self, obs):
    #     def multi_hotcoder(indices):
    #         code = np.zeros(self.num_states, dtype=np.int)
    #         code[indices] = 1
    #         return code
    #     tilecode = self.tilecoder(obs)
    #     hotcode = multi_hotcoder(tilecode)
    #     return hotcode
        
    def random_action(self):
        return self.env.action_space.sample()
        
    def close(self):
        if self.env is not None:
            self.env.close()
            self.env = None
        
    def __del__(self):
        self.close()
            

if __name__ == '__main__':
    env = MountainCar(render=True, tilecoding=True,
                      num_tilings=8, num_tiles_one_dim=8, dim_states=4096)
    
    while True:
        a = env.random_action()
        s_, r, done = env.take_action(a)
        if done:
            break
    
