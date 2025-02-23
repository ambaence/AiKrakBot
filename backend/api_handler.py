import ccxt
import asyncio
import websockets
import json
import logging
from config import MAKER_FEE, TRADING_PAIRS
from backend.security import SecurityManager
from functools import lru_cache
import time

# Structured JSON logging
logging.basicConfig(
    filename='logs/bot.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('logs/bot.log', encoding='utf-8')]
)
logger = logging.getLogger(__name__)

class KrakenAPI:
    def __init__(self, simulate=False):
        """Initialize Kraken API with encrypted key handling and fee caching.

        Args:
            simulate (bool): Whether to run in simulation mode.
        """
        self.simulate = simulate
        self.security = SecurityManager()
        api_key, api_secret = self.security.get_api_credentials()
        self.exchange = ccxt.kraken({
            'apiKey': api_key,
            'secret': api_secret,
            'enableRateLimit': True,
        })
        self.ws_url = "wss://ws.kraken.com"
        self.sim_balance = {'USD': 10000, 'XBT': 0, 'ETH': 0} if simulate else None
        self.logger = logging.getLogger(__name__)
        self.fees = {}
        self.fee_ticks = {}
        self.FEE_CACHE_TICKS = 200
        self.last_call_time = 0
        self.min_call_interval = 0.2  # 5 calls/sec max

    def rate_limit(self):
        """Enforce custom rate limiting beyond ccxt."""
        current_time = time.time()
        elapsed = current_time - self.last_call_time
        if elapsed < self.min_call_interval:
            time.sleep(self.min_call_interval - elapsed)
        self.last_call_time = time.time()

    @lru_cache(maxsize=128)
    def get_balance(self):
        """Fetch or simulate account balance with caching."""
        self.rate_limit()
        self.logger.info({"event": "get_balance", "cached": True})
        try:
            return self.sim_balance if self.simulate else self.exchange.fetch_balance()['total']
        except Exception as e:
            self.logger.error({"event": "get_balance_error", "error": str(e)})
            raise

    def place_order(self, pair, side, amount, price=None):
        """Place a limit order (maker) by default or simulate it."""
        self.rate_limit()
        if self.simulate:
            if price is None:
                price = 10000
            cost = amount * price
            fee = cost * MAKER_FEE
            base, quote = pair.split('/')
            if side == 'buy' and self.sim_balance[quote] >= cost + fee:
                self.sim_balance[quote] -= cost + fee
                self.sim_balance[base] += amount
                self.logger.info({"event": "place_order_simulated", "pair": pair, "side": side, "amount": amount, "price": price})
            elif side == 'sell' and self.sim_balance[base] >= amount:
                self.sim_balance[quote] += cost - fee
                self.sim_balance[base] -= amount
                self.logger.info({"event": "place_order_simulated", "pair": pair, "side": side, "amount": amount, "price": price})
            return {'id': 'sim_order', 'price': price, 'amount': amount, 'side': side}
        else:
            limit_price = price * (1 + 0.001 if side == 'buy' else 1 - 0.001) if price else None
            try:
                order = self.exchange.create_order(
                    pair, 'limit' if limit_price else 'market', side, amount, limit_price
                )
                self.logger.info({"event": "place_order", "pair": pair, "side": side, "amount": amount, "price": limit_price, "order_id": order['id']})
                return order
            except Exception as e:
                self.logger.error({"event": "place_order_error", "pair": pair, "error": str(e)})
                return None

    async def stream_market_data(self, callback):
        """Stream real-time market data via WebSocket with reconnection."""
        retries = 0
        max_retries = 5
        while retries < max_retries:
            try:
                async with websockets.connect(self.ws_url, ping_interval=20, ping_timeout=10) as ws:
                    subscribe_msg = {
                        "event": "subscribe",
                        "pair": TRADING_PAIRS,
                        "subscription": {"name": "ticker"}
                    }
                    await ws.send(json.dumps(subscribe_msg))
                    self.logger.info({"event": "websocket_connected"})
                    retries = 0  # Reset on successful connect
                    while True:
                        try:
                            data = await asyncio.wait_for(ws.recv(), timeout=1.0)
                            data = json.loads(data)
                            if isinstance(data, list):
                                pair = data[3]
                                price = float(data[1]['c'][0])
                                volume = float(data[1]['v'][1])
                                await callback(pair, price, volume)
                        except asyncio.TimeoutError:
                            continue
            except Exception as e:
                retries += 1
                delay = 2 ** retries
                self.logger.error({"event": "websocket_error", "attempt": retries, "max_retries": max_retries, "error": str(e), "retry_delay": delay})
                await asyncio.sleep(delay)
        self.logger.error({"event": "websocket_failed", "reason": "max_retries_exceeded"})

    @lru_cache(maxsize=32)
    def fetch_historical_data(self, pair, timeframe='1m', since=None, limit=1000):
        """Fetch historical OHLCV data for backtesting with caching."""
        self.rate_limit()
        self.logger.info({"event": "fetch_historical_data", "pair": pair, "cached": True})
        try:
            return self.exchange.fetch_ohlcv(pair, timeframe, since, limit)
        except Exception as e:
            self.logger.error({"event": "fetch_historical_data_error", "pair": pair, "error": str(e)})
            return []

    def fetch_trading_fees(self, pair=None):
        """Fetch live trading fees for a specific pair or all pairs, with caching for 200 ticks."""
        if pair not in self.fees or pair not in self.fee_ticks:
            self.fee_ticks[pair] = 0

        if self.fee_ticks.get(pair, 0) < self.FEE_CACHE_TICKS and pair in self.fees:
            self.logger.debug({"event": "fetch_trading_fees", "pair": pair, "cached": True, "fees": self.fees[pair]})
            return self.fees[pair] if pair else self.fees

        self.rate_limit()
        try:
            if self.simulate:
                simulated_fees = {pair or 'BTC/USD': {'maker': MAKER_FEE, 'taker': MAKER_FEE * 1.5}}
                self.logger.info({"event": "fetch_trading_fees_simulated", "pair": pair or "all", "fees": simulated_fees})
                self.fees.update(simulated_fees)
                self.fee_ticks[pair] = 0
                return simulated_fees if pair is None else simulated_fees[pair]
            else:
                fees = self.exchange.fetch_trading_fees()
                if pair:
                    pair_fees = fees.get(pair, {'maker': MAKER_FEE, 'taker': MAKER_FEE * 1.5})
                    self.fees[pair] = pair_fees
                    self.fee_ticks[pair] = 0
                    self.logger.info({"event": "fetch_trading_fees", "pair": pair, "fees": pair_fees})
                    return pair_fees
                else:
                    self.fees.update(fees)
                    for p in fees:
                        self.fee_ticks[p] = 0
                    self.logger.info({"event": "fetch_trading_fees", "pairs": list(fees.keys()), "fees": fees})
                    return fees
        except Exception as e:
            self.logger.error({"event": "fetch_trading_fees_error", "pair": pair or "all", "error": str(e)})
            fallback_fees = {pair or 'BTC/USD': {'maker': MAKER_FEE, 'taker': MAKER_FEE * 1.5}}
            self.fees.update(fallback_fees)
            self.fee_ticks[pair] = 0
            return fallback_fees if pair is None else fallback_fees[pair]

    def increment_fee_tick(self, pair):
        """Increment the tick counter for a specific pair’s fee cache."""
        self.fee_ticks[pair] = self.fee_ticks.get(pair, 0) + 1
        self.logger.debug({"event": "increment_fee_tick", "pair": pair, "ticks": self.fee_ticks[pair]})