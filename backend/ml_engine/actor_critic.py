import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import logging
from config import AC_GAMMA, AC_LR_ACTOR, AC_LR_CRITIC, AC_MEMORY_SIZE, AC_BATCH_SIZE

logging.basicConfig(filename='logs/bot.log', level=logging.INFO)

class ActorCriticModel:
    def __init__(self, state_size=4, action_size=3):
        """Initialize Actor-Critic model for trading decisions."""
        self.state_size = state_size
        self.action_size = action_size
        self.memory = []  # Stores (state, action, reward, next_state, done)
        self.memory_size = AC_MEMORY_SIZE
        self.gamma = AC_GAMMA
        self.actor = self._build_actor()
        self.critic = self._build_critic()
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=AC_LR_ACTOR)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=AC_LR_CRITIC)
        self.trade_size = 0.001
        self.fee_rate = 0.0026
        self.logger = logging.getLogger(__name__)

    def _build_actor(self):
        """Build the Actor network for policy prediction."""
        inputs = layers.Input(shape=(self.state_size,))
        x = layers.Dense(64, activation='relu')(inputs)
        x = layers.Dense(64, activation='relu')(x)
        outputs = layers.Dense(self.action_size, activation='softmax')(x)
        return tf.keras.Model(inputs, outputs, name='actor')

    def _build_critic(self):
        """Build the Critic network for value estimation."""
        inputs = layers.Input(shape=(self.state_size,))
        x = layers.Dense(64, activation='relu')(inputs)
        x = layers.Dense(64, activation='relu')(x)
        outputs = layers.Dense(1)(x)
        return tf.keras.Model(inputs, outputs, name='critic')

    def update_data(self, state, action, reward, next_state, done):
        """Store experience in memory."""
        if len(self.memory) >= self.memory_size:
            self.memory.pop(0)
        self.memory.append((state, action, reward, next_state, done))

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
            profit = (next_price - current_price) * self.trade_size
            return profit - (cost * self.fee_rate)
        elif action == 1:  # Sell
            if position < self.trade_size:
                return -1
            revenue = self.trade_size * current_price * (1 - self.fee_rate)
            return revenue - (current_price * self.trade_size)
        else:  # Hold
            return 0.01 if next_price > current_price else -0.01

    def train(self):
        """Train Actor-Critic using collected experiences."""
        if len(self.memory) < AC_BATCH_SIZE:
            return

        states, actions, rewards, next_states, dones = zip(*self.memory[:AC_BATCH_SIZE])
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_states = np.array(next_states)
        dones = np.array(dones)

        # Compute advantages and returns
        values = self.critic.predict(states, verbose=0).flatten()
        next_values = self.critic.predict(next_states, verbose=0).flatten()
        advantages = rewards + self.gamma * next_values * (1 - dones) - values
        returns = advantages + values

        # Train Actor
        with tf.GradientTape() as tape:
            probs = self.actor(states, training=True)
            action_probs = tf.reduce_sum(probs * tf.one_hot(actions, self.action_size), axis=1)
            actor_loss = -tf.reduce_mean(tf.math.log(action_probs + 1e-10) * advantages)
        actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))

        # Train Critic
        with tf.GradientTape() as tape:
            value_preds = self.critic(states, training=True)
            critic_loss = tf.reduce_mean(tf.square(returns - value_preds))
        critic_grads = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))

        self.memory = self.memory[AC_BATCH_SIZE:]  # Clear trained samples
        self.logger.info("Actor-Critic model trained on batch")

    def predict(self, state):
        """Predict action using the Actor network."""
        state = np.array([state])
        probs = self.actor.predict(state, verbose=0)[0]
        action = np.argmax(probs)
        return action

    def get_price_prediction(self, state):
        """Return current price as dummy prediction for ensemble."""
        return state[0]