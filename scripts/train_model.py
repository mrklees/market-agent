import sys
sys.path.append('.')
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from multiprocessing import Pool, freeze_support
from tqdm import tqdm
from MarketAgent.Market import Market, StockData
from MarketAgent.Trader import ValueTrader
import tensorflow_addons as tfa

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

n_epochs = 100
n_episodes = 100
episode_size = 100
starting_assets = 10000
window_size = 10
memory_size = 1000000
exploration_ratio = 0.3
get_fresh_data = False
train_new_model = True

if get_fresh_data:
    data = StockData(secret_path="./.secrets/alphavantage.key")
    stocks, meta = data.collect_data(
        asset="GOOGL",
        interval='daily',
        output='full'
    )
    stocks.to_csv("./.data/stock_series.csv", index=False)

stocks = pd.read_csv("./.data/stock_series.csv")

market = Market(
    stocks,
    window_size=window_size,
    starting_assets=starting_assets
)
trader = ValueTrader(
    window_size=window_size,
    memory_size=memory_size
)
if train_new_model:
    trader.build_value_network(window_size)
else:
    trader.model = tf.keras.models.load_model("value_10_model.tf")


def process_episode(
    episode,
    trader=trader,
    market=market,
    starting_assets=starting_assets
):
    current_assets = starting_assets
    # Apply the policy to each timestep in the episode, making a
    # -1, 0, 1 decision
    decisions = []
    orders = []
    assets = []
    for row in episode.iterrows():
        # Combine market state with current assets as feature
        # for policy network
        state = np.concatenate([row[1], [current_assets]])
        if random.random() < exploration_ratio:
            # To help with exploration, we'll make random decisions
            # some times during training
            decision = trader.random_action()
        else:
            values = [
                trader.value(proposed_action, state)
                for proposed_action in [-1, 0, 1]
            ]
            decision = trader.policy(values)
        decisions.append(decision)
        # Determine the value of the decision at t_0
        order = decision * row[1][1]
        orders.append(order)
        # Update assets
        new_asset_value = current_assets + order
        assets.append(new_asset_value)
        current_assets = new_asset_value
    episode['assets'] = assets
    episode['decisions'] = decisions
    # If the cumulative sum is greater than zero, than we've sold more
    # stock than we've bought.
    INVALID_SALE_PENALTY = -10000
    validate_sell = np.cumsum(episode['decisions']) > 0
    # If outcomes goes below zero... we're broke
    OUT_OF_MONEY_PENALTY = -10000
    out_of_money = episode['assets'] < 0
    # Net from orders
    net = np.array(orders).sum()
    # Penalties
    sale_penalty = validate_sell * INVALID_SALE_PENALTY
    money_penalty = out_of_money * OUT_OF_MONEY_PENALTY
    reward = net + sale_penalty + money_penalty
    # Save this episode as a emory we can train models from
    episode = np.concatenate(
        [
            episode,
            reward.values.reshape(-1, 1)
        ],
        axis=1
    )
    return episode


if __name__ == "__main__":
    freeze_support()
    pool = Pool(6)
    for epoch in tqdm(range(n_epochs)):
        #print(f"Starting Epock {epoch}")
        episodes = [
            market.get_episode(episode_size=episode_size)
            for i in range(n_episodes)
        ]
        # Each episode is independent, so we process them in parallel
        memories = pool.map(
            process_episode,
            episodes
        )
        trader.memory.extend(list(memories))
        trader.process_memory(window_size)
