import logging
import numpy as np
import pandas as pd
from config import TRADING_PAIRS, MAX_TRADE_SIZE, RISK_TOLERANCE, MAX_PORTFOLIO_RISK, STOP_LOSS_MULTIPLIER, LOOKBACK_PERIOD

class RiskManager:
    def __init__(self, api):
        """Initialize risk manager with API and settings."""
        self.api = api
        self.max_portfolio_risk = MAX_PORTFOLIO_RISK
        self.positions = {pair: 0 for pair in TRADING_PAIRS}
        self.entry_prices = {pair: None for pair in TRADING_PAIRS}
        self.volatility = {pair: 0.01 for pair in TRADING_PAIRS}
        self.price_history = {pair: [] for pair in TRADING_PAIRS}
        logging.basicConfig(filename='logs/bot.log', level=logging.INFO)

    def update_position(self, pair, amount, price, side):
        """Update position and entry price after a trade."""
        if side == 'buy':
            self.positions[pair] += amount
            if self.entry_prices[pair] is None:
                self.entry_prices[pair] = price
            else:
                total_cost = self.entry_prices[pair] * (self.positions[pair] - amount) + price * amount
                self.entry_prices[pair] = total_cost / self.positions[pair]
        elif side == 'sell':
            self.positions[pair] -= amount
            if self.positions[pair] <= 0:
                self.entry_prices[pair] = None

    def update_volatility(self, pair, price):
        """Update price history and volatility."""
        self.price_history[pair].append(price)
        if len(self.price_history[pair]) > LOOKBACK_PERIOD:
            self.price_history[pair].pop(0)
        if len(self.price_history[pair]) >= LOOKBACK_PERIOD:
            prices = self.price_history[pair]
            mean = sum(prices) / len(prices)
            variance = sum((p - mean) ** 2 for p in prices) / len(prices)
            self.volatility[pair] = (variance ** 0.5) / mean

    def check_stop_loss_take_profit(self, pair, current_price):
        """Check and execute dynamic stop-loss or take-profit."""
        if self.positions[pair] <= 0 or self.entry_prices[pair] is None:
            return None

        entry_price = self.entry_prices[pair]
        stop_loss = entry_price * (1 - RISK_TOLERANCE * STOP_LOSS_MULTIPLIER * (1 + self.volatility[pair]))
        take_profit = entry_price * (1 + RISK_TOLERANCE * 2 * (1 + self.volatility[pair]))

        if current_price <= stop_loss:
            amount = self.positions[pair]
            self.api.place_order(pair, 'sell', amount, current_price)
            self.positions[pair] = 0
            self.entry_prices[pair] = None
            logging.info(f"Stop-loss triggered: Sold {amount} {pair} at {current_price}")
            return 'stop_loss'
        elif current_price >= take_profit:
            amount = self.positions[pair]
            self.api.place_order(pair, 'sell', amount, current_price)
            self.positions[pair] = 0
            self.entry_prices[pair] = None
            logging.info(f"Take-profit triggered: Sold {amount} {pair} at {current_price}")
            return 'take_profit'
        return None

    def calculate_risk_parity(self, balance):
        """Risk parity using covariance matrix."""
        total_usd = balance.get('USD', 0) + sum(
            self.positions[pair] * self.price_history[pair][-1]
            for pair in TRADING_PAIRS if self.price_history[pair]
        )
        if total_usd == 0:
            return {pair: MAX_TRADE_SIZE for pair in TRADING_PAIRS}

        price_data = {pair: self.price_history[pair] for pair in TRADING_PAIRS if len(self.price_history[pair]) >= LOOKBACK_PERIOD}
        if len(price_data) < 2:
            risk_weights = {pair: 1 / self.volatility[pair] if self.volatility[pair] > 0 else 1 for pair in TRADING_PAIRS}
            total_weight = sum(risk_weights.values())
            return {pair: (weight / total_weight) * total_usd / self.price_history[pair][-1] for pair, weight in risk_weights.items() if self.price_history[pair]}

        df = pd.DataFrame(price_data)
        returns = df.pct_change().dropna()
        if len(returns) < LOOKBACK_PERIOD // 2:
            logging.warning("Insufficient return data for covariance; using fallback")
            return {pair: MAX_TRADE_SIZE for pair in TRADING_PAIRS}

        cov_matrix = returns.cov().values
        if np.linalg.matrix_rank(cov_matrix) < len(price_data):
            logging.warning("Covariance matrix singular; using diagonal fallback")
            volatilities = np.sqrt(np.diag(cov_matrix))
        else:
            volatilities = np.sqrt(np.diag(cov_matrix))

        risk_weights = 1 / volatilities
        total_weight = sum(risk_weights)
        allocations = {}
        for i, pair in enumerate(price_data.keys()):
            usd_allocation = (risk_weights[i] / total_weight) * total_usd
            allocations[pair] = min(MAX_TRADE_SIZE, usd_allocation / self.price_history[pair][-1]) if self.price_history[pair] else MAX_TRADE_SIZE
        return allocations

    def monitor_portfolio_risk(self, balance):
        """Check if portfolio risk exceeds max limit."""
        total_value = balance.get('USD', 0)
        unrealized_pnl = 0
        for pair in TRADING_PAIRS:
            if self.positions[pair] > 0 and self.price_history[pair]:
                current_price = self.price_history[pair][-1]
                total_value += self.positions[pair] * current_price
                unrealized_pnl += self.positions[pair] * (current_price - self.entry_prices[pair])

        if total_value > 0:
            portfolio_risk = -unrealized_pnl / total_value if unrealized_pnl < 0 else 0
            if portfolio_risk > self.max_portfolio_risk:
                logging.warning(f"Portfolio risk {portfolio_risk:.2%} exceeds limit {self.max_portfolio_risk:.2%}")
                return False, portfolio_risk
        return True, portfolio_risk if total_value > 0 else 0

    def stress_test(self, pair, current_price):
        """Simulate a 10% price drop to estimate max loss."""
        if self.positions[pair] > 0 and self.entry_prices[pair]:
            simulated_price = current_price * 0.9
            loss = self.positions[pair] * (simulated_price - self.entry_prices[pair])
            return loss
        return 0