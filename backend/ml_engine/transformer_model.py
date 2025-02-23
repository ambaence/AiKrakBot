import numpy as np
import pandas as pd
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, MultiHeadAttention, LayerNormalization, Dropout

class TransformerModel:
    def __init__(self, lookback=20, num_heads=4, ff_dim=32):
        """Transformer model for multi-scale dependencies."""
        self.lookback = lookback
        self.data = []
        inputs = Input(shape=(lookback, 2))
        attn_output = MultiHeadAttention(num_heads=num_heads, key_dim=2)(inputs, inputs)
        attn_output = Dropout(0.1)(attn_output)
        norm1 = LayerNormalization(epsilon=1e-6)(inputs + attn_output)
        ff_output = Dense(ff_dim, activation='relu')(norm1)
        ff_output = Dense(2)(ff_output)
        norm2 = LayerNormalization(epsilon=1e-6)(norm1 + ff_output)
        output = Dense(1)(norm2[:, -1, :])
        self.model = Model(inputs, output)
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
        """Train the Transformer model."""
        X, y = self.prepare_data()
        if X is not None:
            self.model.fit(X, y, epochs=1, batch_size=32, verbose=0)

    def predict(self):
        """Predict the next price."""
        X, _ = self.prepare_data()
        if X is not None:
            return self.model.predict(X[-1:], verbose=0)[0][0]
        return None