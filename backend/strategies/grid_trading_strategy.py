from config import MAX_TRADE_SIZE, RISK_TOLERANCE, TRADING_PAIRS, GRID_LEVELS, GRID_SPACING
import logging
import numpy as np

class GridTradingStrategy:
    def __init__(self, api, model, risk_manager):
        """Initialize Grid Trading strategy."""
        self.api = api
        self.model = model
        self.risk_manager = risk_manager
        self.positions = {pair: 0 for pair in TRADING_PAIRS}
        self.grid_orders = {pair: [] for pair in TRADING_PAIRS}  # List of (price, amount, side) tuples
        self.logger = logging.getLogger(__name__)

    def execute(self, pair, current_price, predicted_price, action):
        """Execute Grid Trading strategy."""
        balance = self.api.get_balance()
        usd = balance.get('USD', 0)
        base = 'XBT' if pair.startswith('BTC') else 'ETH'
        crypto = balance.get(base, 0)

        self.risk_manager.update_volatility(pair, current_price)
        if self.risk_manager.check_stop_loss_take_profit(pair, current_price):
            return
        if not self.risk_manager.monitor_portfolio_risk(balance)[0]:
            self.logger.info("Portfolio risk too high; Grid Trading aborted")
            return

        # Clear executed orders and update positions
        executed_orders = []
        for order in self.grid_orders[pair]:
            price, amount, side = order
            if (side == 'buy' and current_price <= price and usd >= amount * price) or \
               (side == 'sell' and current_price >= price and crypto >= amount):
                executed_orders.append(order)
                if side == 'buy':
                    self.api.place_order(pair, 'buy', amount, price)
                    self.risk_manager.update_position(pair, amount, price, 'buy')
                    self.positions[pair] += amount
                    self.logger.info(f"Grid Buy: {amount} {pair} at {price}")
                elif side == 'sell':
                    self.api.place_order(pair, 'sell', amount, price)
                    self.risk_manager.update_position(pair, amount, price, 'sell')
                    self.positions[pair] -= amount
                    self.logger.info(f"Grid Sell: {amount} {pair} at {price}")
        self.grid_orders[pair] = [o for o in self.grid_orders[pair] if o not in executed_orders]

        # Place new grid orders if none exist
        if not self.grid_orders[pair]:
            allocations = self.risk_manager.calculate_risk_parity(balance)
            max_amount = min(MAX_TRADE_SIZE, allocations.get(pair, MAX_TRADE_SIZE))
            grid_amount = max_amount / GRID_LEVELS

            for i in range(-GRID_LEVELS, GRID_LEVELS + 1):
                grid_price = current_price * (1 + i * GRID_SPACING)
                if i < 0 and usd >= grid_amount * grid_price:  # Buy orders below current price
                    self.grid_orders[pair].append((grid_price, grid_amount, 'buy'))
                elif i > 0 and crypto >= grid_amount:  # Sell orders above current price
                    self.grid_orders[pair].append((grid_price, grid_amount, 'sell'))
            self.logger.info(f"Placed {len(self.grid_orders[pair])} grid orders for {pair}")