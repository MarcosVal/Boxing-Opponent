import numpy as np


class ReplayBuffer(object):
    def __init__(self, max_size=1000000):
        self.storage = []
        self.max_size = max_size
        self.ptr = 0

    def __len__(self):
        return len(self.storage)

    def add(self, data):
        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)] = data
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append(data)

    def sample(self, batch_size):
        ind = np.random.randint(0, len(self.storage), size=batch_size)
        state, action, next_state, reward, done = [], [], [], [], []

        for i in ind:
            s, a, s_, r, d = self.storage[i]
            state.append(np.array(s, copy=False))
            action.append(np.array(a, copy=False))
            next_state.append(np.array(s_, copy=False))
            reward.append(r)
            done.append(d)

        return np.array(state), np.array(action), np.array(next_state), \
            np.array(reward).reshape(-1, 1), np.array(done).reshape(-1, 1)
