from config import MAX_TRADE_SIZE, RISK_TOLERANCE, TRADING_PAIRS
import logging

class ScalpingStrategy:
    def __init__(self, api, model, risk_manager):
        """Initialize scalping strategy."""
        self.api = api
        self.model = model
        self.risk_manager = risk_manager
        self.positions = {pair: 0 for pair in TRADING_PAIRS}

    def execute(self, pair, current_price, predicted_price, action):
        """Execute scalping trades with tight thresholds."""
        balance = self.api.get_balance()
        usd = balance.get('USD', 0)
        btc = balance.get('XBT', 0) if pair.startswith('BTC') else balance.get('ETH', 0)

        self.risk_manager.update_volatility(pair, current_price)
        if self.risk_manager.check_stop_loss_take_profit(pair, current_price):
            return
        if not self.risk_manager.monitor_portfolio_risk(balance)[0]:
            return

        thresh = RISK_TOLERANCE / 2
        allocations = self.risk_manager.calculate_risk_parity(balance)
        max_amount = min(MAX_TRADE_SIZE, allocations.get(pair, MAX_TRADE_SIZE))

        if predicted_price > current_price * (1 + thresh) and usd > 10:
            amount = min(max_amount, usd / current_price)
            self.api.place_order(pair, 'buy', amount, current_price)
            self.risk_manager.update_position(pair, amount, current_price, 'buy')
            self.positions[pair] += amount
            logging.info(f"Scalping buy: {amount} {pair} at {current_price}")
        elif predicted_price < current_price * (1 - thresh) and self.positions[pair] > 0:
            amount = min(max_amount, self.positions[pair])
            self.api.place_order(pair, 'sell', amount, current_price)
            self.risk_manager.update_position(pair, amount, current_price, 'sell')
            self.positions[pair] -= amount
            logging.info(f"Scalping sell: {amount} {pair} at {current_price}")