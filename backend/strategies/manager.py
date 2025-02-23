from .momentum_strategy import MomentumStrategy
from .scalping_strategy import ScalpingStrategy
from .arbitrage_strategy import ArbitrageStrategy
from .mean_reversion_strategy import MeanReversionStrategy
from .pair_trading_strategy import PairTradingStrategy
from .breakout_strategy import BreakoutStrategy
from .grid_trading_strategy import GridTradingStrategy
from .dca_strategy import DCAStrategy
import numpy as np
from config import TRADING_PAIRS, LOOKBACK_PERIOD, PAIR_CORR_THRESHOLD

class StrategyManager:
    def __init__(self, api, model, risk_manager):
        """Initialize strategy manager with expanded strategies."""
        self.api = api
        self.model = model
        self.risk_manager = risk_manager
        self.strategies = {
            'momentum': MomentumStrategy(api, model, risk_manager),
            'scalping': ScalpingStrategy(api, model, risk_manager),
            'arbitrage': ArbitrageStrategy(api, risk_manager),
            'mean_reversion': MeanReversionStrategy(api, model, risk_manager),
            'pair_trading': PairTradingStrategy(api, model, risk_manager),
            'breakout': BreakoutStrategy(api, model, risk_manager),
            'grid_trading': GridTradingStrategy(api, model, risk_manager),
            'dca': DCAStrategy(api, model, risk_manager)
        }
        self.current_strategy = 'momentum'
        self.price_history = {pair: [] for pair in TRADING_PAIRS}

    def update(self, pair, price):
        """Detect market conditions and switch strategies."""
        self.price_history[pair].append(price)
        if len(self.price_history[pair]) > LOOKBACK_PERIOD:
            self.price_history[pair].pop(0)
            prices = self.price_history[pair]
            volatility = np.std(prices) / np.mean(prices)
            if volatility > 0.02:
                self.current_strategy = 'scalping'
            elif len(self.strategies['arbitrage'].prices) >= 3:
                self.current_strategy = 'arbitrage'
            elif volatility < 0.01:
                self.current_strategy = 'mean_reversion'
            elif pair in ["BTC/USD", "ETH/USD"] and len(self.price_history["BTC/USD"]) >= LOOKBACK_PERIOD and len(self.price_history["ETH/USD"]) >= LOOKBACK_PERIOD:
                corr = np.corrcoef(self.price_history["BTC/USD"], self.price_history["ETH/USD"])[0, 1]
                if corr > PAIR_CORR_THRESHOLD:
                    self.current_strategy = 'pair_trading'
            elif volatility > 0.015:
                self.current_strategy = 'breakout'
            elif volatility < 0.005:  # Low volatility favors Grid Trading
                self.current_strategy = 'grid_trading'
            elif volatility < 0.01:  # Stable markets favor DCA
                self.current_strategy = 'dca'
            else:
                self.current_strategy = 'momentum'

    def execute(self, pair, price, predicted_price, action):
        """Execute the current strategy."""
        if self.current_strategy == 'arbitrage':
            self.strategies['arbitrage'].update_prices(pair, price)
            self.strategies['arbitrage'].execute()
        elif self.current_strategy == 'pair_trading':
            self.strategies['pair_trading'].update_prices(pair, price)
            self.strategies['pair_trading'].execute(pair, price, predicted_price, action)
        else:
            self.strategies[self.current_strategy].execute(pair, price, predicted_price, action)