from config import TRADING_PAIRS, MAX_TRADE_SIZE, TAKER_FEE
import logging

class ArbitrageStrategy:
    def __init__(self, api, risk_manager):
        """Initialize arbitrage strategy."""
        self.api = api
        self.risk_manager = risk_manager
        self.prices = {}

    def update_prices(self, pair, price):
        """Update price data for arbitrage detection."""
        self.prices[pair] = price
        self.risk_manager.update_volatility(pair, price)

    def execute(self):
        """Detect and execute triangular arbitrage."""
        balance = self.api.get_balance()
        if not self.risk_manager.monitor_portfolio_risk(balance)[0]:
            return
        if len(self.prices) < 3:
            return

        btc_usd = self.prices.get("BTC/USD")
        eth_usd = self.prices.get("ETH/USD")
        eth_btc = self.prices.get("ETH/BTC")
        if not all([btc_usd, eth_usd, eth_btc]):
            return

        usd_start = min(100, balance.get('USD', 0))
        btc_amount = usd_start / btc_usd
        eth_amount = btc_amount / eth_btc
        usd_end = eth_amount * eth_usd
        profit = usd_end - usd_start - (usd_start * TAKER_FEE * 3)

        if profit > 1:
            amount = min(MAX_TRADE_SIZE, usd_start / btc_usd)
            self.api.place_order("BTC/USD", 'buy', amount, btc_usd)
            self.api.place_order("ETH/BTC", 'buy', amount / eth_btc, eth_btc)
            self.api.place_order("ETH/USD", 'sell', amount / eth_btc, eth_usd)
            logging.info(f"Arbitrage profit: ${profit:.2f}")