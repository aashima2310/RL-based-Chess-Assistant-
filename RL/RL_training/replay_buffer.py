import random
import numpy as np
from collections import deque

class ReplayBuffer:

    def __init__(self, max_size=50000):
        self.buffer = deque(maxlen=max_size)

    def push(self, data):
        self.buffer.extend(data)


    def sample(self, batch_size):

        batch = random.sample(self.buffer, batch_size)
        states, policies, values = zip(*batch)

        return states, np.array(policies), np.array(values, dtype=np.float32)

    def __len__(self):
        return len(self.buffer)

