import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense

class GRUModel:
    def __init__(self, lookback=20):
        """GRU model for faster sequence processing."""
        self.lookback = lookback
        self.data = []
        self.model = Sequential([
            GRU(50, return_sequences=True, input_shape=(lookback, 2)),
            GRU(50),
            Dense(1)
        ])
        self.model.compile(optimizer='adam', loss='mse')

    def update_data(self, price, volume):
        """Add new price and volume data."""
        self.data.append([price, volume])
        if len(self.data) > 1000:
            self.data.pop(0)

    def prepare_data(self):
        """Prepare data for training/prediction."""
        if len(self.data) < self.lookback + 1:
            return None, None
        df = pd.DataFrame(self.data, columns=['price', 'volume'])
        X, y = [], []
        for i in range(len(df) - self.lookback):
            X.append(df.iloc[i:i+self.lookback].values)
            y.append(df.iloc[i+self.lookback]['price'])
        return np.array(X), np.array(y)

    def train(self):
        """Train the GRU model."""
        X, y = self.prepare_data()
        if X is not None:
            self.model.fit(X, y, epochs=1, batch_size=32, verbose=0)

    def predict(self):
        """Predict the next price."""
        X, _ = self.prepare_data()
        if X is not None:
            return self.model.predict(X[-1:], verbose=0)[0][0]
        return None