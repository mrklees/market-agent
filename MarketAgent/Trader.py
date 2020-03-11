from collections import deque
import numpy as np


class Trader(object):
    def __init__(self, memory_size=10000):
        self.possible_actions = ['buy', 'sell', 'hold']
        self.policy_net = None
        self.value_net = None
        self.memory = deque(maxlen=memory_size)

    def act(self):
        raise NotImplementedError

    def policy(self, state):
        raise NotImplementedError

    def value(self, state):
        raise NotImplementedError


class RandomTrader(Trader):
    def __init__(self):
        super().__init__()

    def act(self):
        probs = [0.3, 0.3, 0.4]
        return np.random.choice(self.possible_actions, p=probs)
