import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Dense, Dropout

class TCNModel:
    def __init__(self, lookback=20, filters=32, kernel_size=3):
        """TCN for high-frequency data analysis."""
        self.lookback = lookback
        self.data = []
        self.model = Sequential([
            Conv1D(filters=filters, kernel_size=kernel_size, padding='causal', activation='relu', input_shape=(lookback, 2)),
            Conv1D(filters=filters, kernel_size=kernel_size, padding='causal', activation='relu'),
            Dropout(0.2),
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
        X = []
        for i in range(len(df) - self.lookback):
            X.append(df.iloc[i:i+self.lookback].values)
        y = df['price'][self.lookback:].values
        return np.array(X), np.array(y)

    def train(self):
        """Train the TCN model."""
        X, y = self.prepare_data()
        if X is not None:
            self.model.fit(X, y, epochs=1, batch_size=32, verbose=0)

    def predict(self):
        """Predict the next price."""
        X, _ = self.prepare_data()
        if X is not None:
            return self.model.predict(X[-1:], verbose=0)[0][0]
        return None