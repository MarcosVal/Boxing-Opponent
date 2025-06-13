import gym
from gym.wrappers import AtariPreprocessing, FrameStack
import torch
import numpy as np
from tqdm import tqdm
from wrappers import ClipReward
from replay_buffer import ReplayBuffer
from models.dqn import DQN
import os

class Trainer(object):
    def __init__(self, config):
        self.config = config
        self.env = gym.make(config['env'])  # , difficulty=3)

        # pre-processing techniques
        self.env = AtariPreprocessing(self.env, scale_obs=True)
        self.env = ClipReward(self.env, -1, 1)
        self.env = FrameStack(self.env, num_stack=4)

        # device
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # dimensions going in and results of network
        out_dim = self.env.action_space.n
        in_dim = self.env.observation_space.shape[0]

        # taking lives lost into consideration for Breakout
        # self.lives = env.ale.lives()


        # change DQN to DDQN or D3QN  TODO: models parser
        self.agent = DQN(in_dim, out_dim,
                         config['lr'], config['gamma'],
                         config['epsilon_min'], config['epsilon_decay'],
                         config['batch_size'], device)

        # pick memory size
        self.replay_buffer = ReplayBuffer(10000)

        # save path
        self.raw_dir = "./results/raw/" + self.config['env']
        self.model_dir = "./results/models/" + self.config['env']
        if not os.path.exists(self.raw_dir):
            os.makedirs(self.raw_dir)
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)


    def train(self):
        total_step = 0
        progress = tqdm(range(self.config['episodes']))
        rewards = []

        for episode in progress:
            state, i = self.env.reset()
            ep_reward = 0

            while True:
                total_step += 1
                action = self.agent.act_with_noise(state)
                next_state, reward, terminated, truncated, i = self.env.step(action)
                self.replay_buffer.add((state, action, next_state, reward, terminated or truncated))
                state = next_state
                ep_reward += reward

                # # breakout lives
                # current_lives = info['lives']
                # if current_lives < self.lives:
                #     total_reward = total_reward - 1
                #     self.lives = current_lives

                if total_step > self.config['warmup'] and total_step % self.config['update_freq'] == 0:
                    self.agent.update(self.replay_buffer)
                if total_step > self.config['warmup'] and total_step % self.config['update_target'] == 0:
                    self.agent.update_target()
                if terminated or truncated:
                    break

            progress.set_description('Episode: {}/{} | Episode Reward: {:.2f}'.
                                     format(episode+1, self.config['episodes'], ep_reward))
            rewards.append(ep_reward)


        # save rewards
        np.save(self.raw_dir + "/dqn.npy", rewards)
        #self.agent.save(self.model_dir + "/dqn.pt")

        torch.save(self.agent.Q.stat_dict(), self.model_dir + "/dqn.pt")
