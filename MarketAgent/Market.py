import random
import numpy as np
import pandas as pd
from alpha_vantage.timeseries import TimeSeries


class StockData(object):
    def __init__(self,
                 assets=['GOOGL'],
                 secret_path="./.secrets/alphavantage.key"):
        with open(secret_path) as f:
            self.key = f.read()

    def collect_data(self, asset='GOOGL', interval='1min', output='compact'):
        ts = TimeSeries(key=self.key, output_format='pandas')
        if interval in ['1min', '1min', '5min', '15min', '30min', '60min']:
            data, meta_data = ts.get_intraday(
                asset,
                interval=output,
                outputsize=output
            )
        elif interval in ['daily']:
            data, meta_data = ts.get_daily(
                asset,
                outputsize=output
            )
        else:
            raise ValueError("Please use an appropriate interval in '1min', \
                              '1min', '5min', '15min', '30min', '60min' or \
                              'daily' ")

        series = (
            data
                .reset_index()[['date', '2. high']]
                .rename({"2. high": "t_0"}, axis=1)
                .drop(['date'], axis=1)
        )
        return series, meta_data


class Market(object):
    def __init__(
        self,
        series,
        window_size=5,
        starting_assets=1000
    ):
        self.series = self.add_window(series.copy(), window_size)
        self.assets = starting_assets
        self.window = window_size
        self.t = 0
        self.orders = np.zeros(self.series.size)

    def add_window(self, series, window_size):
        series = pd.DataFrame(series, columns=["t_0"])
        for period in range(1, window_size + 1):
            series.insert(
                len(series.columns), f"t_{period}", series['t_0'].shift(period)
            )
        series = series.dropna()
        return series

    def get_episode(self, episode_size=100):
        max_ix = self.series.shape[0] - episode_size
        random_ix = int(random.random() * max_ix)
        return self.series.iloc[random_ix:random_ix + episode_size]

    def step(self, action=0, volume=1):
        state = self.episodes.iloc[self.t].values

        # if action == -1, buy
        # if action == 0, hold
        # if action == 1, sell
        self.orders[self.t] = action * (state[self.window - 1]*volume)

        reward = np.sum(
            self.orders[~np.isnan(self.orders)][max(0, self.t - 100):self.t]
        )

        self.t += 1
        next_state = self.episodes.iloc[self.t].values

        if self.t == self.episodes.shape[0] - 1:
            done = True
        else:
            done = False

        return np.concatenate(
            [np.array([np.sum(self.orders[~np.isnan(self.orders)])]),
             next_state],
            axis=0
        ), reward, done
