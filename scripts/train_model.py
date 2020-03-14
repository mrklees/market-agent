import sys
sys.path.append('.')
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from MarketAgent.Market import Market, StockData
from MarketAgent.Trader import ValueTrader

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

n_epochs = 25
n_episodes = 200
episode_size = 100
starting_assets = 10000
window_size = 10
memory_size = 100000

data = StockData(secret_path="./.secrets/alphavantage.key")
stocks, meta = data.collect_data(
    asset="GOOGL",
    interval='daily',
    output='full'
)
market = Market(
    stocks,
    window_size=window_size,
    starting_assets=starting_assets
)
trader = ValueTrader(window_size=window_size, memory_size=memory_size)
# trader.build_value_network(window_size)
trader.model = tf.keras.models.load_model("value_10_model.tf")

for epoch in range(n_epochs):
    print(f"Training new epoch of {n_episodes} episodes")
    for episode in tqdm(range(n_episodes)):
        episode = market.get_episode(episode_size=episode_size)
        # Apply the policy to each timestep in the episode, making a
        # -1, 0, 1 decision
        decisions = []
        orders = []
        assets = []
        for row in episode.iterrows():
            # Combine market state with current assets as feature
            # for policy network
            state = np.concatenate([row[1], [market.assets]])
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
            new_asset_value = market.assets + order
            assets.append(new_asset_value)
            market.assets = new_asset_value
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
        trader.memory.append(episode)
        # Reset market
        market.assets = starting_assets
    print("Updating Model")
    trader.process_memory(window_size)
