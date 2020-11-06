from env.MountainCar import MountainCar
from env.CartPole import CartPole
from agents.QLearning import QLearning
from agents.Sarsa import SemiGradSarsa
from agents.dqn import DeepQLearning as DQN


# tilecoding is one type of feature engineering
# for linear policy and linear function approximation,
# whith is suited for SemiGradient-Sarsa and Q-learning(with Q-table).
# for neural network approximator, the input could be the raw state data.
# So, what is left is:
# 0. save model for each algorithm
# 1. replace the raw state version of DQN
# 2. transfer to CartPole-v0 and v1
# 3. Read the ERL cpt 9-11 and DQN paper
# 4. Try Double-DQN and go up towards Rainbow


if __name__ == '__main__':
    # # Algo 1: Semi-Gradient Sarsa
    # env = MountainCar(num_tilings=8, num_tiles_one_dim=8, num_states=4096)
    # leaner = SemiGradSarsa(env)
    # leaner.train(num_eposide=200)
    # env.close()
    
    # # Algo 2:
    # env = MountainCar(num_tilings=1, num_tiles_one_dim=32, num_states=10000)
    # leaner = QLearning(env)
    # leaner.train(num_eposide=10000)
    # env.close()
    
    # # Algo 3: Deep Q-Learning
    # env = MountainCar(num_tilings=8, num_tiles_one_dim=8, num_states=512, render=False)
    # leaner = DQN(env)
    # leaner.train(num_eposide=1000)
    # env.close()
    
    # Algo 3: Deep Q-Learning
    env = CartPole(render=False)
    leaner = DQN(env)
    leaner.train(num_eposide=1000)
    env.close()
    
