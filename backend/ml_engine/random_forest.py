import numpy as np
from sklearn.ensemble import RandomForestRegressor
import shap
import logging
from config import (
    RF_N_ESTIMATORS, RF_MAX_DEPTH, RF_MIN_SAMPLES_SPLIT, LOOKBACK_PERIOD,
    RF_RSI_PERIOD, RF_MACD_FAST, RF_MACD_SLOW, RF_MACD_SIGNAL,
    RF_BB_PERIOD, RF_BB_STD, RF_EMA_SHORT, RF_EMA_LONG, RF_MOMENTUM_PERIOD
)

logging.basicConfig(filename='logs/bot.log', level=logging.INFO)

class RandomForestModel:
    def __init__(self):
        """Initialize Random Forest model with advanced features and SHAP support."""
        self.model = RandomForestRegressor(
            n_estimators=RF_N_ESTIMATORS,
            max_depth=RF_MAX_DEPTH,
            min_samples_split=RF_MIN_SAMPLES_SPLIT,
            random_state=42
        )
        self.data_buffer = []  # List of [price, volume] pairs
        self.logger = logging.getLogger(__name__)
        self.is_trained = False
        self.explainer = None
        self.last_X = None  # Store last training features for SHAP

    def update_data(self, price, volume):
        """Add new data to buffer."""
        max_period = max(LOOKBACK_PERIOD, RF_RSI_PERIOD, RF_MACD_SLOW, RF_BB_PERIOD, RF_EMA_LONG, RF_MOMENTUM_PERIOD)
        self.data_buffer.append([price, volume])
        if len(self.data_buffer) > max_period + 1:
            self.data_buffer.pop(0)

    def _calculate_rsi(self, prices):
        """Calculate RSI over a given period."""
        if len(prices) < RF_RSI_PERIOD + 1:
            return 50.0  # Neutral default
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        avg_gain = np.mean(gains[-RF_RSI_PERIOD:])
        avg_loss = np.mean(losses[-RF_RSI_PERIOD:])
        rs = avg_gain / avg_loss if avg_loss != 0 else float('inf')
        rsi = 100 - (100 / (1 + rs)) if rs != float('inf') else 100
        return rsi

    def _calculate_macd(self, prices):
        """Calculate MACD, MACD Signal, and MACD Histogram."""
        if len(prices) < RF_MACD_SLOW:
            return 0.0, 0.0, 0.0
        ema_fast = self._calculate_ema(prices, RF_MACD_FAST)
        ema_slow = self._calculate_ema(prices, RF_MACD_SLOW)
        macd = ema_fast - ema_slow
        signal = self._calculate_ema(np.array([macd] * len(prices))[-RF_MACD_SIGNAL:], RF_MACD_SIGNAL) if len(prices) >= RF_MACD_SIGNAL else 0
        hist = macd - signal
        return macd, signal, hist

    def _calculate_ema(self, prices, period):
        """Calculate Exponential Moving Average."""
        if len(prices) < period:
            return np.mean(prices) if prices.size > 0 else 0.0
        alpha = 2 / (period + 1)
        ema = prices[0]
        for price in prices[1:]:
            ema = alpha * price + (1 - alpha) * ema
        return ema

    def _calculate_bollinger_bands(self, prices):
        """Calculate Bollinger Bands (upper, middle, lower)."""
        if len(prices) < RF_BB_PERIOD:
            return prices[-1], prices[-1], prices[-1] if prices.size > 0 else (0, 0, 0)
        sma = np.mean(prices[-RF_BB_PERIOD:])
        std = np.std(prices[-RF_BB_PERIOD:])
        upper = sma + RF_BB_STD * std
        lower = sma - RF_BB_STD * std
        return upper, sma, lower

    def _extract_features(self):
        """Extract advanced features from data buffer for Random Forest."""
        if len(self.data_buffer) < LOOKBACK_PERIOD:
            return None, None
        data = np.array(self.data_buffer[-LOOKBACK_PERIOD:])
        prices = data[:, 0]
        volumes = data[:, 1]

        price_mean = np.mean(prices)
        volume_mean = np.mean(volumes)
        price_volatility = np.std(prices)
        volume_volatility = np.std(volumes)
        price_trend = (prices[-1] - prices[0]) / LOOKBACK_PERIOD if LOOKBACK_PERIOD > 0 else 0

        full_prices = np.array([d[0] for d in self.data_buffer])
        rsi = self._calculate_rsi(full_prices[-min(len(full_prices), RF_RSI_PERIOD + 1):])
        macd, macd_signal, macd_hist = self._calculate_macd(full_prices[-min(len(full_prices), RF_MACD_SLOW):])
        bb_upper, bb_middle, bb_lower = self._calculate_bollinger_bands(full_prices[-min(len(full_prices), RF_BB_PERIOD):])
        ema_short = self._calculate_ema(full_prices[-min(len(full_prices), RF_EMA_SHORT):], RF_EMA_SHORT)
        ema_long = self._calculate_ema(full_prices[-min(len(full_prices), RF_EMA_LONG):], RF_EMA_LONG)
        momentum = full_prices[-1] - full_prices[-RF_MOMENTUM_PERIOD] if len(full_prices) >= RF_MOMENTUM_PERIOD else 0

        features = [
            price_mean, volume_mean, price_volatility, volume_volatility, price_trend,
            rsi, macd, macd_signal, macd_hist, bb_upper, bb_middle, bb_lower,
            ema_short, ema_long, momentum
        ]
        target = prices[-1]
        return np.array(features).reshape(1, -1), target

    def train(self):
        """Train Random Forest on historical data with advanced features and compute SHAP values."""
        required_length = max(LOOKBACK_PERIOD + 1, RF_RSI_PERIOD + 1, RF_MACD_SLOW, RF_BB_PERIOD, RF_EMA_LONG, RF_MOMENTUM_PERIOD)
        if len(self.data_buffer) < required_length:
            return

        X, y = [], []
        for i in range(len(self.data_buffer) - LOOKBACK_PERIOD):
            window = self.data_buffer[i:i+LOOKBACK_PERIOD]
            prices = [d[0] for d in window]
            volumes = [d[1] for d in window]
            
            price_mean = np.mean(prices)
            volume_mean = np.mean(volumes)
            price_volatility = np.std(prices)
            volume_volatility = np.std(volumes)
            price_trend = (prices[-1] - prices[0]) / LOOKBACK_PERIOD if LOOKBACK_PERIOD > 0 else 0
            
            full_prices = np.array([d[0] for d in self.data_buffer[:i+LOOKBACK_PERIOD]])
            rsi = self._calculate_rsi(full_prices[-min(len(full_prices), RF_RSI_PERIOD + 1):])
            macd, macd_signal, macd_hist = self._calculate_macd(full_prices[-min(len(full_prices), RF_MACD_SLOW):])
            bb_upper, bb_middle, bb_lower = self._calculate_bollinger_bands(full_prices[-min(len(full_prices), RF_BB_PERIOD):])
            ema_short = self._calculate_ema(full_prices[-min(len(full_prices), RF_EMA_SHORT):], RF_EMA_SHORT)
            ema_long = self._calculate_ema(full_prices[-min(len(full_prices), RF_EMA_LONG):], RF_EMA_LONG)
            momentum = full_prices[-1] - full_prices[-RF_MOMENTUM_PERIOD] if len(full_prices) >= RF_MOMENTUM_PERIOD else 0

            X.append([
                price_mean, volume_mean, price_volatility, volume_volatility, price_trend,
                rsi, macd, macd_signal, macd_hist, bb_upper, bb_middle, bb_lower,
                ema_short, ema_long, momentum
            ])
            y.append(self.data_buffer[i+LOOKBACK_PERIOD][0])

        X = np.array(X)
        y = np.array(y)
        
        if len(X) > 0:
            self.model.fit(X, y)
            self.is_trained = True
            self.last_X = X  # Store for SHAP
            self.explainer = shap.TreeExplainer(self.model)
            shap_values = self.explainer.shap_values(X[-1:])
            self.logger.info(f"Random Forest trained on {len(X)} samples with advanced features. SHAP values: {shap_values[0]}")
            return shap_values[0]  # Return SHAP values for last sample

    def get_shap_values(self):
        """Get SHAP values for the last trained data point."""
        if self.is_trained and self.last_X is not None and self.explainer is not None:
            return self.explainer.shap_values(self.last_X[-1:])
        return None

    def predict(self):
        """Predict next price using Random Forest with advanced features."""
        if not self.is_trained or len(self.data_buffer) < LOOKBACK_PERIOD:
            return None
        features, _ = self._extract_features()
        return float(self.model.predict(features)[0])

if __name__ == "__main__":
    rf_model = RandomForestModel()
    for i in range(30):
        rf_model.update_data(10000 + i * 10, 10 + i)
    rf_model.train()
    pred = rf_model.predict()
    print(f"Predicted Price: {pred}")