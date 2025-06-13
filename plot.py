import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    plt.figure(dpi=300)

    # adapt for env
    smooth = 100
    dqn = np.convolve(np.load('results/raw/BoxingNoFrameskip-v4/dqn.npy'), np.ones(smooth)/smooth, mode='valid')
    ddqn = np.convolve(np.load('results/raw/BoxingNoFrameskip-v4/ddqn.npy'), np.ones(smooth)/smooth, mode='valid')
    d3qn = np.convolve(np.load('results/raw/BoxingNoFrameskip-v4/d3qn.npy'), np.ones(smooth)/smooth, mode='valid')

    plt.plot(dqn, label='DQN')
    plt.plot(ddqn, label='DDQN')
    plt.plot(d3qn, label='D3QN')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()

    # graph name
    plt.savefig('results/graphs/BoxingNoFrameskip-v4.png')
    