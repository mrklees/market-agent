import numpy as np


class StockData(object):
    def __init__(self):
        with open("./.secrets/alphavantage.key") as f:
            self.key = f.read()


class Market(object):
    def __init__(self, data, starting_assets=1000, window=5):
        self.data = data
        self.assets = starting_assets
        self.window = window
        self.t = window
        self.orders = np.zeros(self.data.size)
        self.possible_actions = ['buy', 'sell', 'hold']

    def step(self, action='hold', volume=1):
        # supports action in ['buy', 'sell', 'hold']
        if action not in self.possible_actions:
            raise ValueError("Invalid input")

        time, state = self.state()
        if action == 'buy':
            self.orders[time] = -(state[self.window - 1]*volume)
        if action == 'sell':
            self.orders[time] = state[self.window - 1]*volume

        reward = np.sum(
            self.orders[~np.isnan(self.orders)][max(0, self.t-100):self.t]
        )

        self.t += 1
        next_time, next_state = self.state()

        if self.t == self.data.size - 1:
            done = True
        else:
            done = False

        return np.concatenate(
            [np.array([np.sum(self.orders[~np.isnan(self.orders)])]),
             next_state],
            axis=0
        ), reward, done

    def state(self):
        return self.t, self.data[self.t-5:self.t]
