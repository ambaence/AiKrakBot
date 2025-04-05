import pandas as pd
import numpy as np
from backend.api_handler import KrakenAPI
from backend.ml_engine.ensemble import EnsembleModel
from backend.strategies.manager import StrategyManager
import logging

class Backtester:
    def __init__(self, api, model, strategy_manager):
        """Initialize backtester."""
        self.api = api
        self.model = model
        self.strategy_manager = strategy_manager
        self.initial_balance = self.api.get_balance()['USD']

    def run(self, pair, start_date=None, end_date=None, use_synthetic=False):
        """Run backtest on historical or tuned synthetic data."""
        if use_synthetic:
            # Use tuned GAN-generated synthetic data
            synthetic_data = self.model.generate_synthetic_batch(num_samples=1000)
            data = []
            for sample in synthetic_data:
                price = sample[-1, 0]  # Last price in sequence
                volume = sample[-1, 1]  # Last volume in sequence
                data.append([0, 0, 0, 0, price, volume])  # Dummy OHLC
        else:
            # Fetch historical data
            data = self.api.fetch_historical_data(pair, since=start_date, limit=1000)
        
        df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        for _, row in df.iterrows():
            price, volume = row['close'], row['volume']
            self.model.update_data(price, volume, self.api.get_balance(), 0)
            self.model.train()
            prediction = self.model.predict(self.api.get_balance(), 0)
            if prediction['price']:
                self.strategy_manager.update(pair, price)
                self.strategy_manager.execute(pair, price, prediction['price'], prediction['action'])
        final_balance = self.api.get_balance()['USD']
        profit = final_balance - self.initial_balance
        logging.info(f"Backtest profit for {pair} (synthetic={use_synthetic}): ${profit:.2f}")
        return profit
