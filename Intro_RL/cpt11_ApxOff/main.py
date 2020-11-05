from env.MountainCar import MountainCar
from agents.QLearning import QLearning
from agents.Sarsa import SemiGradSarsa
from agents.dqn import DeepQNetwork as DQN


if __name__ == '__main__':
    # Algo 1: Semi-Gradient Sarsa
    env = MountainCar(num_tilings=8, num_tiles_one_dim=8, num_states=4096)
    leaner = SemiGradSarsa(env)
    leaner.train(num_eposide=200)
    env.close()
    
    # # Algo 2:
    # env = MountainCar(num_tilings=1, num_tiles_one_dim=32, num_states=10000)
    # leaner = QLearning(env)
    # leaner.train(num_eposide=10000)
    # env.close()
    
    # # Algo 3: Deep Q-Learning
    # env = MountainCar(num_tilings=8, num_tiles_one_dim=8, num_states=512)
    # leaner = DQN(env)
    # # leaner.train(num_eposide=200)
    # env.close()