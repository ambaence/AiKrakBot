import numpy as np
import tensorflow as tf
from collections import deque
import random

class DQNModel:
    def __init__(self, state_size=4, action_size=3, memory_size=2000, gamma=0.95, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995):
        """
        DQN Model for trading decisions.
        - State: [price, volume, balance_USD, position]
        - Actions: [buy, sell, hold]
        """
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=memory_size)  # Experience replay buffer
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.model = self._build_model()

    def _build_model(self):
        """Build the Q-network."""
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(24, input_dim=self.state_size, activation='relu'),
            tf.keras.layers.Dense(24, activation='relu'),
            tf.keras.layers.Dense(self.action_size, activation='linear')
        ])
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
        return model

    def update_data(self, state, action, reward, next_state, done):
        """Store experience in memory."""
        self.memory.append((state, action, reward, next_state, done))

    def train(self, batch_size=32):
        """Train the DQN model using experience replay."""
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        states = np.array([t[0] for t in minibatch])
        actions = np.array([t[1] for t in minibatch])
        rewards = np.array([t[2] for t in minibatch])
        next_states = np.array([t[3] for t in minibatch])
        dones = np.array([t[4] for t in minibatch])

        targets = self.model.predict(states, verbose=0)
        next_q_values = self.model.predict(next_states, verbose=0)
        for i in range(batch_size):
            targets[i][actions[i]] = rewards[i] + self.gamma * np.max(next_q_values[i]) * (1 - dones[i])

        self.model.fit(states, targets, epochs=1, verbose=0)

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def predict(self, state):
        """Predict action based on current state."""
        if random.random() <= self.epsilon:
            return random.randrange(self.action_size)  # Explore
        q_values = self.model.predict(np.array([state]), verbose=0)
        return np.argmax(q_values[0])  # Exploit

    def get_price_prediction(self, state):
        """For compatibility with other models, return a dummy price prediction."""
        return state[0]  # Return current price as baseline