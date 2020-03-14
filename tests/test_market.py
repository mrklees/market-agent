import numpy as np
from MarketAgent.Market import Market


def test_market_step():
    market = Market(np.random.random(size=100), window=5)
    s, r, done = market.step()
    assert s.shape[0] == 7
    assert not done


def test_market_step_through():
    market = Market(np.random.randint(0, 2, size=100))
    s, r, done = market.step()
    while not done:
        s, r, done = market.step()
