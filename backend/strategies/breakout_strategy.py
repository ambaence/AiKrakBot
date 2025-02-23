from config import MAX_TRADE_SIZE, RISK_TOLERANCE, TRADING_PAIRS, BREAKOUT_THRESHOLD, LOOKBACK_PERIOD
import logging
import numpy as np

class BreakoutStrategy:
    def __init__(self, api, model, risk_manager):
        """Initialize Breakout strategy."""
        self.api = api
        self.model = model
        self.risk_manager = risk_manager
        self.positions = {pair: 0 for pair in TRADING_PAIRS}
        self.logger = logging.getLogger(__name__)

    def execute(self, pair, current_price, predicted_price, action):
        """Execute Breakout trades based on support/resistance levels."""
        balance = self.api.get_balance()
        usd = balance.get('USD', 0)
        btc = balance.get('XBT', 0) if pair.startswith('BTC') else balance.get('ETH', 0)

        self.risk_manager.update_volatility(pair, current_price)
        if self.risk_manager.check_stop_loss_take_profit(pair, current_price):
            return
        if not self.risk_manager.monitor_portfolio_risk(balance)[0]:
            self.logger.info("Portfolio risk too high; Breakout aborted")
            return

        price_history = self.risk_manager.price_history.get(pair, [])
        if len(price_history) < LOOKBACK_PERIOD:
            return

        resistance = max(price_history[-LOOKBACK_PERIOD:])
        support = min(price_history[-LOOKBACK_PERIOD:])
        thresh = BREAKOUT_THRESHOLD * self.risk_manager.volatility[pair]
        allocations = self.risk_manager.calculate_risk_parity(balance)
        max_amount = min(MAX_TRADE_SIZE, allocations.get(pair, MAX_TRADE_SIZE))

        if current_price > resistance * (1 + thresh) and usd > 10:
            amount = min(max_amount, usd / current_price)
            if amount > 0:
                self.api.place_order(pair, 'buy', amount, current_price)
                self.risk_manager.update_position(pair, amount, current_price, 'buy')
                self.positions[pair] += amount
                self.logger.info(f"Breakout Buy: {amount} {pair} at {current_price} (Resistance: {resistance})")
        elif current_price < support * (1 - thresh) and self.positions[pair] > 0:
            amount = min(max_amount, self.positions[pair])
            if amount > 0:
                self.api.place_order(pair, 'sell', amount, current_price)
                self.risk_manager.update_position(pair, amount, current_price, 'sell')
                self.positions[pair] -= amount
                self.logger.info(f"Breakout Sell: {amount} {pair} at {current_price} (Support: {support})")