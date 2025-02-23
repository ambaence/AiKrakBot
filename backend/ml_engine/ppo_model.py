import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import logging
from config import PPO_GAMMA, PPO_CLIP, PPO_LR, PPO_MEMORY_SIZE, PPO_BATCH_SIZE

# Setup logging
logging.basicConfig(filename='logs/bot.log', level=logging.INFO)

class PPOModel:
    def __init__(self, state_size=4, action_size=3):
        """Initialize PPO model for trading decisions."""
        self.state_size = state_size
        self.action_size = action_size
        self.memory = []  # List for storing (state, action, reward, next_state, done, prob)
        self.memory_size = PPO_MEMORY_SIZE
        self.gamma = PPO_GAMMA
        self.clip = PPO_CLIP
        self.actor = self._build_actor()
        self.critic = self._build_critic()
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=PPO_LR)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=PPO_LR)
        self.last_action = None
        self.last_price = None
        self.trade_size = 0.001  # Simulated trade size (BTC/ETH)
        self.fee_rate = 0.0026  # Kraken taker fee (0.26%)
        self.logger = logging.getLogger(__name__)

    def _build_actor(self):
        """Build the actor network for policy prediction."""
        inputs = layers.Input(shape=(self.state_size,))
        x = layers.Dense(64, activation='relu')(inputs)
        x = layers.Dense(64, activation='relu')(x)
        outputs = layers.Dense(self.action_size, activation='softmax')(x)
        return tf.keras.Model(inputs, outputs, name='actor')

    def _build_critic(self):
        """Build the critic network for value estimation."""
        inputs = layers.Input(shape=(self.state_size,))
        x = layers.Dense(64, activation='relu')(inputs)
        x = layers.Dense(64, activation='relu')(x)
        outputs = layers.Dense(1)(x)
        return tf.keras.Model(inputs, outputs, name='critic')

    def update_data(self, state, action, reward, next_state, done, prob):
        """Store experience in memory."""
        if len(self.memory) >= self.memory_size:
            self.memory.pop(0)
        self.memory.append((state, action, reward, next_state, done, prob))
        self.last_action = action
        self.last_price = state[0]

    def calculate_reward(self, state, next_state, action):
        """Calculate reward based on profit/loss with fees."""
        current_price = state[0]
        next_price = next_state[0]
        usd = state[2]
        position = state[3]

        if action == 0:  # Buy
            cost = self.trade_size * current_price * (1 + self.fee_rate)
            if usd < cost:
                return -1
            potential_profit = (next_price - current_price) * self.trade_size
            reward = potential_profit - (cost * self.fee_rate)
        elif action == 1:  # Sell
            if position < self.trade_size:
                return -1
            revenue = self.trade_size * current_price * (1 - self.fee_rate)
            if self.last_action == 0:
                profit = revenue - (self.last_price * self.trade_size * (1 + self.fee_rate))
                reward = profit
            else:
                reward = revenue - (current_price * self.trade_size)
        else:  # Hold
            reward = 0.01 if next_price > current_price else -0.01
        return reward

    def train(self):
        """Train the PPO model using collected experiences."""
        if len(self.memory) < PPO_BATCH_SIZE:
            return

        # Convert memory to arrays
        states, actions, rewards, next_states, dones, old_probs = zip(*self.memory)
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_states = np.array(next_states)
        dones = np.array(dones)
        old_probs = np.array(old_probs)

        # Compute advantages and returns
        values = self.critic.predict(states, verbose=0).flatten()
        next_values = self.critic.predict(next_states, verbose=0).flatten()
        advantages = rewards + self.gamma * next_values * (1 - dones) - values
        returns = advantages + values

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Train actor
        with tf.GradientTape() as tape:
            probs = self.actor(states, training=True)
            action_probs = tf.reduce_sum(probs * tf.one_hot(actions, self.action_size), axis=1)
            ratio = action_probs / (old_probs + 1e-10)
            clipped_ratio = tf.clip_by_value(ratio, 1 - self.clip, 1 + self.clip)
            actor_loss = -tf.reduce_mean(tf.minimum(ratio * advantages, clipped_ratio * advantages))
        actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))

        # Train critic
        with tf.GradientTape() as tape:
            value_preds = self.critic(states, training=True)
            critic_loss = tf.reduce_mean(tf.square(returns - value_preds))
        critic_grads = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))

        self.memory = []  # Clear memory after training
        self.logger.info("PPO model trained on batch")

    def predict(self, state):
        """Predict action and probability using the actor network."""
        state = np.array([state])
        probs = self.actor.predict(state, verbose=0)[0]
        action = np.argmax(probs)
        return action, probs[action]

    def get_price_prediction(self, state):
        """Return current price as a dummy prediction for ensemble compatibility."""
        return state[0]