import gym
import time

#导入MountainCar-v0环境
env = gym.make('MountainCar-v0')
#初始化环境
env.reset()
#循环1000次
for _ in range(1000):
    #绘图
    env.render()
    #进行一个动作
    info = env.step(env.action_space.sample()) # take a random action
    
    print(info)
    # 慢动作展示
    time.sleep(0.001)
#关闭
env.close()