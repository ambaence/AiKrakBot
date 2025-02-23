from config import MAX_TRADE_SIZE, RISK_TOLERANCE, TRADING_PAIRS
import logging
import numpy as np

class MeanReversionStrategy:
    def __init__(self, api, model, risk_manager):
        """Initialize mean reversion strategy."""
        self.api = api
        self.model = model
        self.risk_manager = risk_manager
        self.positions = {pair: 0 for pair in TRADING_PAIRS}
        logging.basicConfig(filename='logs/bot.log', level=logging.INFO)

    def execute(self, pair, current_price, predicted_price, action):
        """Execute mean reversion trades based on historical average."""
        balance = self.api.get_balance()
        usd = balance.get('USD', 0)
        btc = balance.get('XBT', 0) if pair.startswith('BTC') else balance.get('ETH', 0)

        self.risk_manager.update_volatility(pair, current_price)
        if self.risk_manager.check_stop_loss_take_profit(pair, current_price):
            return
        if not self.risk_manager.monitor_portfolio_risk(balance)[0]:
            logging.info("Portfolio risk too high; trade aborted")
            return

        price_history = self.risk_manager.price_history.get(pair, [])
        if len(price_history) < 20:
            return
        sma = np.mean(price_history[-20:])
        thresh = RISK_TOLERANCE * (1 + self.risk_manager.volatility[pair])

        allocations = self.risk_manager.calculate_risk_parity(balance)
        max_amount = min(MAX_TRADE_SIZE, allocations.get(pair, MAX_TRADE_SIZE))

        if current_price < sma * (1 - thresh) and usd > 10:
            amount = min(max_amount, usd / current_price)
            if amount > 0:
                self.api.place_order(pair, 'buy', amount, current_price)
                self.risk_manager.update_position(pair, amount, current_price, 'buy')
                self.positions[pair] += amount
                logging.info(f"Mean reversion buy: {amount} {pair} at {current_price} (SMA: {sma})")
        elif current_price > sma * (1 + thresh) and self.positions[pair] > 0:
            amount = min(max_amount, self.positions[pair])
            if amount > 0:
                self.api.place_order(pair, 'sell', amount, current_price)
                self.risk_manager.update_position(pair, amount, current_price, 'sell')
                self.positions[pair] -= amount
                logging.info(f"Mean reversion sell: {amount} {pair} at {current_price} (SMA: {sma})")