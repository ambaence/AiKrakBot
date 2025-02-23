from config import MAX_TRADE_SIZE, RISK_TOLERANCE, TRADING_PAIRS
import logging

class MomentumStrategy:
    def __init__(self, api, model):
        self.api = api
        self.model = model
        self.positions = {pair: 0 for pair in TRADING_PAIRS}

    def execute(self, pair, current_price, predicted_price, action):
        balance = self.api.get_balance()
        usd = balance.get('USD', 0)
        btc = balance.get('XBT', 0)

        if action == 0 or (predicted_price > current_price * (1 + RISK_TOLERANCE) and usd > 10):  # Buy
            amount = min(MAX_TRADE_SIZE, usd / current_price)
            self.api.place_order(pair, 'buy', amount, current_price)
            self.positions[pair] += amount
            logging.info(f"Bought {amount} {pair} at {current_price}")
        elif action == 1 or (predicted_price < current_price * (1 - RISK_TOLERANCE) and btc > 0):  # Sell
            amount = min(MAX_TRADE_SIZE, btc)
            self.api.place_order(pair, 'sell', amount, current_price)
            self.positions[pair] -= amount
            logging.info(f"Sold {amount} {pair} at {current_price}")
        # Action 2 (hold) does nothing