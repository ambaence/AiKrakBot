import numpy as np
import tensorflow as tf
from spektral.layers import GCNConv
import logging
from config import GNN_LR, GNN_HIDDEN_DIM, GNN_LAYERS, GNN_CHANNELS, LOOKBACK_PERIOD

logging.basicConfig(filename='logs/bot.log', level=logging.INFO)

class GNNModel:
    def __init__(self, input_shape=(LOOKBACK_PERIOD, 2)):
        """Initialize advanced GNN model for market data analysis."""
        self.input_shape = input_shape
        self.hidden_dim = GNN_HIDDEN_DIM
        self.layers = GNN_LAYERS
        self.channels = GNN_CHANNELS
        self.model = self._build_gnn()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=GNN_LR)
        self.data_buffer = []
        self.logger = logging.getLogger(__name__)

    def _build_gnn(self):
        """Build a GNN using Graph Convolutional Networks."""
        # Input: node features (price, volume over time) and adjacency matrix
        node_input = tf.keras.Input(shape=self.input_shape, name='node_features')  # (batch, LOOKBACK_PERIOD, 2)
        adj_input = tf.keras.Input(shape=(LOOKBACK_PERIOD, LOOKBACK_PERIOD), name='adjacency_matrix')  # (batch, LOOKBACK_PERIOD, LOOKBACK_PERIOD)
        
        # Graph Convolutional Layers
        x = node_input
        for _ in range(self.layers):
            x = GCNConv(self.channels, activation='relu')([x, adj_input])
        
        # Global pooling to reduce graph to single prediction
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
        output = tf.keras.layers.Dense(1, activation='linear')(x)  # Predict next price

        model = tf.keras.Model(inputs=[node_input, adj_input], outputs=output)
        model.compile(optimizer=self.optimizer, loss='mse')
        return model

    def update_data(self, price, volume):
        """Add new data to buffer."""
        self.data_buffer.append([price, volume])
        if len(self.data_buffer) > LOOKBACK_PERIOD:
            self.data_buffer.pop(0)

    def train(self):
        """Train GNN on recent data with constructed graph."""
        if len(self.data_buffer) < LOOKBACK_PERIOD:
            return
        
        # Construct node features and adjacency matrix
        node_features = np.array(self.data_buffer).reshape(1, LOOKBACK_PERIOD, 2)
        adj_matrix = self._create_adjacency_matrix()  # Simple fully-connected graph for now
        target = np.array([node_features[0, -1, 0]])  # Predict last price as target
        
        self.model.train_on_batch([node_features, adj_matrix], target)
        self.logger.info("GNN trained on recent data")

    def _create_adjacency_matrix(self):
        """Create a simple adjacency matrix (fully connected)."""
        adj = np.ones((LOOKBACK_PERIOD, LOOKBACK_PERIOD)) - np.eye(LOOKBACK_PERIOD)
        return adj.reshape(1, LOOKBACK_PERIOD, LOOKBACK_PERIOD)

    def predict(self):
        """Predict next price using GNN."""
        if len(self.data_buffer) < LOOKBACK_PERIOD:
            return None
        node_features = np.array(self.data_buffer).reshape(1, LOOKBACK_PERIOD, 2)
        adj_matrix = self._create_adjacency_matrix()
        return float(self.model.predict([node_features, adj_matrix], verbose=0)[0])