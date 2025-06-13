import torch
import gym
from gym.wrappers import AtariPreprocessing, FrameStack
from models.dqn import DQN
from wrappers import ClipReward

class Tester(object):
    def __init__(self, config):
        # create env
        self.config = config
        self.env = gym.make(config['env'], render_mode="human")
        self.env = AtariPreprocessing(self.env, scale_obs=True)
        self.env = FrameStack(self.env, num_stack=4)  # test
        self.env = ClipReward(self.env, -1, 1)

        # params
        self.c = self.env.observation_space.shape[0]
        self.h = self.env.observation_space.shape[1]
        self.w = self.env.observation_space.shape[2]
        action_dim = self.env.action_space.n
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        lr = self.config['lr']
        gamma = self.config['gamma']
        batch_size = self.config['batch_size']
        
        # model
        self.agent = DQN(self.in_channels, self.h, self.w, action_dim, lr, gamma,
                          config['epsilon_min'], config['epsilon_decay'], batch_size, device)
        self.agent.load("./results/models/" + self.config['env'] + "/dqn.pt")


    def test(self):
        self.agent.Q.eval()
        self.agent.Q_target.eval()

        rewards = []
        for i in range(20):
            state, i = self.env.reset()
            ep_reward = 0
            while True:
                action = self.agent.act(state)
                next_state, reward, terminated, truncated, i = self.env.step(
                    action)
                state = next_state
                ep_reward += reward
                if terminated or truncated:
                    break
            rewards.append(ep_reward)
            print(f'Episode score: {ep_reward}')
