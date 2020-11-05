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
        
    def hotcoder(self, obs):
        def multi_hotcoder(indices):
            code = np.zeros(self.num_states, dtype=np.int)
            code[indices] = 1
        hotcodes = []
        for i, item in enumerate(obs):
            tilecode = self.tilecoder(obs)
            hotcode = multi_hotcoder(tilecode)
            hotcodes.append(hotcode)
        return hotcodes

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
            

if __name__ == '__main__':
    env = MountainCar(num_tilings=8, num_tiles_one_dim=8, num_states=4096)
    s = env.get_state()
    tilecode = env.tilecoder(s)
    
