import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import logging
from config import AE_LATENT_DIM, AE_LR, AE_ANOMALY_THRESHOLD, LOOKBACK_PERIOD

logging.basicConfig(filename='logs/bot.log', level=logging.INFO)

class AutoencoderModel:
    def __init__(self, input_shape=(LOOKBACK_PERIOD, 2)):
        """Initialize Autoencoder for anomaly detection."""
        self.input_shape = input_shape
        self.latent_dim = AE_LATENT_DIM
        self.model = self._build_autoencoder()
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=AE_LR), loss='mse')
        self.data_buffer = []
        self.anomaly_threshold = AE_ANOMALY_THRESHOLD
        self.logger = logging.getLogger(__name__)

    def _build_autoencoder(self):
        """Build the Autoencoder network."""
        inputs = layers.Input(shape=self.input_shape)
        # Encoder
        x = layers.Flatten()(inputs)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dense(64, activation='relu')(x)
        encoded = layers.Dense(self.latent_dim)(x)
        # Decoder
        x = layers.Dense(64, activation='relu')(encoded)
        x = layers.Dense(128, activation='relu')(x)
        decoded = layers.Dense(self.input_shape[0] * self.input_shape[1], activation='linear')(x)
        decoded = layers.Reshape(self.input_shape)(decoded)
        return tf.keras.Model(inputs, decoded, name='autoencoder')

    def update_data(self, price, volume):
        """Add new data to buffer."""
        self.data_buffer.append([price, volume])
        if len(self.data_buffer) > 1000:
            self.data_buffer.pop(0)

    def train(self):
        """Train the Autoencoder on recent data."""
        if len(self.data_buffer) < LOOKBACK_PERIOD + 1:
            return
        data = np.array([self.data_buffer[i:i+LOOKBACK_PERIOD] 
                        for i in range(len(self.data_buffer) - LOOKBACK_PERIOD)])
        self.model.fit(data, data, epochs=1, batch_size=32, verbose=0)
        self.logger.info("Autoencoder trained on recent data")

    def detect_anomaly(self, state):
        """Detect anomalies in the latest market state."""
        if len(self.data_buffer) < LOOKBACK_PERIOD:
            return False
        recent_data = np.array(self.data_buffer[-LOOKBACK_PERIOD:]).reshape(1, *self.input_shape)
        recon = self.model.predict(recent_data, verbose=0)
        mse = np.mean((recent_data - recon) ** 2)
        is_anomaly = mse > self.anomaly_threshold
        if is_anomaly:
            self.logger.warning(f"Anomaly detected: MSE={mse:.4f} exceeds threshold {self.anomaly_threshold}")
        return is_anomaly