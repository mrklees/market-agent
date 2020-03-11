import numpy as np
from MarketAgent.Market import Market


def test_market_step():
    market = Market(np.random.random(size=100))
    s, r, done = market.step()
    assert s.shape[0] == 6
    assert not done
