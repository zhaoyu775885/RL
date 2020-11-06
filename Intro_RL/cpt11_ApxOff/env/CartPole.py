import gym
import numpy as np

def run_demo():
    env = gym.make('MountainCar-v0') #导入MountainCar-v0环境
    env.reset() #初始化环境
    for _ in range(100): #循环100次
        env.render() #绘图
        a = env.action_space.sample() #进行一个动作
        info = env.step(a) # take a random action
    env.close() #关闭
    
class CartPole():
    def __init__(self, render=True):
        self.env = gym.make('CartPole-v0')
        self.default_render = render
        self.num_actions = self.env.action_space.n
        self.num_states = self.env.observation_space.shape[0]
        self.state = self.init()
        
    def init(self):
        obs = self.env.reset()
        if self.default_render:
            self.env.render()
        return obs
    
    def hotcoder(self, obs):
        return obs
    
    def get_state(self):
        return self.state
        
    def random_action(self):
        return self.env.action_space.sample()
        
    def take_action(self, a):
        s, r, status, info = self.env.step(a)
        if self.default_render:
            self.env.render()
        self.state = s
        return s, r, status

    def close(self):
        if self.env is not None:
            self.env.close()
            self.env = None
        
    def __del__(self):
        self.close()
            

if __name__ == '__main__':
    env = CartPole(render=True)
    
    for _ in range(10):
        print('restart')
        env.init()
        while True:
            a = env.random_action()
            s_, r, done = env.take_action(a)  
            if done:
                break
        
    env.close()
