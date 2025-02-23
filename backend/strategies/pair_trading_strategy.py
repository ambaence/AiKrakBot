from config import MAX_TRADE_SIZE, RISK_TOLERANCE, TRADING_PAIRS, PAIR_CORR_THRESHOLD, LOOKBACK_PERIOD
import logging
import numpy as np

class PairTradingStrategy:
    def __init__(self, api, model, risk_manager):
        """Initialize Pair Trading strategy for BTC/USD and ETH/USD."""
        self.api = api
        self.model = model
        self.risk_manager = risk_manager
        self.positions = {pair: 0 for pair in TRADING_PAIRS}
        self.price_history = {"BTC/USD": [], "ETH/USD": []}
        self.logger = logging.getLogger(__name__)

    def update_prices(self, pair, price):
        """Update price history for pair trading pairs."""
        if pair in self.price_history:
            self.price_history[pair].append(price)
            if len(self.price_history[pair]) > LOOKBACK_PERIOD:
                self.price_history[pair].pop(0)

    def execute(self, pair, current_price, predicted_price, action):
        """Execute Pair Trading based on BTC/USD and ETH/USD correlation."""
        balance = self.api.get_balance()
        usd = balance.get('USD', 0)
        btc = balance.get('XBT', 0)
        eth = balance.get('ETH', 0)

        self.risk_manager.update_volatility(pair, current_price)
        if self.risk_manager.check_stop_loss_take_profit(pair, current_price):
            return
        if not self.risk_manager.monitor_portfolio_risk(balance)[0]:
            self.logger.info("Portfolio risk too high; Pair Trading aborted")
            return

        if len(self.price_history["BTC/USD"]) < LOOKBACK_PERIOD or len(self.price_history["ETH/USD"]) < LOOKBACK_PERIOD:
            return

        btc_prices = np.array(self.price_history["BTC/USD"])
        eth_prices = np.array(self.price_history["ETH/USD"])
        corr = np.corrcoef(btc_prices, eth_prices)[0, 1]
        if corr < PAIR_CORR_THRESHOLD:
            return

        spread = btc_prices[-1] - eth_prices[-1] * (np.mean(btc_prices) / np.mean(eth_prices))
        mean_spread = np.mean(btc_prices - eth_prices * (np.mean(btc_prices) / np.mean(eth_prices)))
        std_spread = np.std(btc_prices - eth_prices * (np.mean(btc_prices) / np.mean(eth_prices)))

        allocations = self.risk_manager.calculate_risk_parity(balance)
        max_amount = min(MAX_TRADE_SIZE, allocations.get(pair, MAX_TRADE_SIZE))

        if spread > mean_spread + 2 * std_spread and usd > 10:
            btc_amount = min(max_amount, btc)
            eth_amount = min(max_amount, usd / eth_prices[-1])
            if btc_amount > 0 and eth_amount > 0:
                self.api.place_order("BTC/USD", 'sell', btc_amount, btc_prices[-1])
                self.api.place_order("ETH/USD", 'buy', eth_amount, eth_prices[-1])
                self.risk_manager.update_position("BTC/USD", btc_amount, btc_prices[-1], 'sell')
                self.risk_manager.update_position("ETH/USD", eth_amount, eth_prices[-1], 'buy')
                self.positions["BTC/USD"] -= btc_amount
                self.positions["ETH/USD"] += eth_amount
                self.logger.info(f"Pair Trade: Sold {btc_amount} BTC/USD, Bought {eth_amount} ETH/USD")
        elif spread < mean_spread - 2 * std_spread and usd > 10:
            btc_amount = min(max_amount, usd / btc_prices[-1])
            eth_amount = min(max_amount, eth)
            if btc_amount > 0 and eth_amount > 0:
                self.api.place_order("BTC/USD", 'buy', btc_amount, btc_prices[-1])
                self.api.place_order("ETH/USD", 'sell', eth_amount, eth_prices[-1])
                self.risk_manager.update_position("BTC/USD", btc_amount, btc_prices[-1], 'buy')
                self.risk_manager.update_position("ETH/USD", eth_amount, eth_prices[-1], 'sell')
                self.positions["BTC/USD"] += btc_amount
                self.positions["ETH/USD"] -= eth_amount
                self.logger.info(f"Pair Trade: Bought {btc_amount} BTC/USD, Sold {eth_amount} ETH/USD")