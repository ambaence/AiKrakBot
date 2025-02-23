from config import MAX_TRADE_SIZE, RISK_TOLERANCE, TRADING_PAIRS, DCA_AMOUNT, DCA_INTERVAL
import logging
import time

class DCAStrategy:
    def __init__(self, api, model, risk_manager):
        """Initialize Dollar-Cost Averaging strategy."""
        self.api = api
        self.model = model
        self.risk_manager = risk_manager
        self.positions = {pair: 0 for pair in TRADING_PAIRS}
        self.last_buy_time = {pair: 0 for pair in TRADING_PAIRS}
        self.logger = logging.getLogger(__name__)

    def execute(self, pair, current_price, predicted_price, action):
        """Execute DCA strategy with fixed USD buy amounts."""
        balance = self.api.get_balance()
        usd = balance.get('USD', 0)

        self.risk_manager.update_volatility(pair, current_price)
        if self.risk_manager.check_stop_loss_take_profit(pair, current_price):
            return
        if not self.risk_manager.monitor_portfolio_risk(balance)[0]:
            self.logger.info("Portfolio risk too high; DCA aborted")
            return

        current_time = time.time()
        if current_time - self.last_buy_time[pair] >= DCA_INTERVAL and usd >= DCA_AMOUNT:
            amount = min(MAX_TRADE_SIZE, DCA_AMOUNT / current_price)
            if amount > 0:
                self.api.place_order(pair, 'buy', amount, current_price)
                self.risk_manager.update_position(pair, amount, current_price, 'buy')
                self.positions[pair] += amount
                self.last_buy_time[pair] = current_time
                self.logger.info(f"DCA Buy: {amount} {pair} at {current_price} for ${DCA_AMOUNT}")