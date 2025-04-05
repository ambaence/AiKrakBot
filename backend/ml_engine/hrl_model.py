import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
import logging
from config import TRADING_PAIRS

logging.basicConfig(
    filename='logs/bot.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('logs/bot.log', encoding='utf-8')]
)
logger = logging.getLogger(__name__)

class HRLModel:
    def __init__(self, state_size=4, action_size=3, buffer_size=10000, batch_size=32, gamma=0.99, tau=0.001):
        """Initialize Hierarchical RL with meta-controller and sub-policies."""
        self.state_size = state_size
        self.action_size = action_size
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = logging.getLogger(__name__)
        self.logger.info({"event": "hrl_init", "device": str(self.device), "state_size": state_size, "action_size": action_size})

        self.meta_controller = nn.Sequential(
            nn.Linear(state_size + len(TRADING_PAIRS), 64),
            nn.ReLU(),
            nn.Linear(64, len(TRADING_PAIRS)),
            nn.Softmax(dim=-1)
        ).to(self.device)
        self.meta_target = nn.Sequential(
            nn.Linear(state_size + len(TRADING_PAIRS), 64),
            nn.ReLU(),
            nn.Linear(64, len(TRADING_PAIRS)),
            nn.Softmax(dim=-1)
        ).to(self.device)
        self.meta_target.load_state_dict(self.meta_controller.state_dict())

        self.sub_policy = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.ReLU(),
            nn.Linear(64, action_size),
            nn.Softmax(dim=-1)
        ).to(self.device)
        self.sub_target = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.ReLU(),
            nn.Linear(64, action_size),
            nn.Softmax(dim=-1)
        ).to(self.device)
        self.sub_target.load_state_dict(self.sub_policy.state_dict())

        self.meta_optimizer = torch.optim.Adam(self.meta_controller.parameters(), lr=0.0003)
        self.sub_optimizer = torch.optim.Adam(self.sub_policy.parameters(), lr=0.0003)
        
        self.memory = deque(maxlen=buffer_size)

    # ... rest of the methods (store_experience, predict, train) ...

    def store_experience(self, state, action, reward, next_state, done):
        """Store experience in the replay buffer."""
        try:
            if not (isinstance(state, list) and len(state) == self.state_size and 
                    isinstance(next_state, list) and len(next_state) == self.state_size):
                raise ValueError("Invalid state or next_state dimensions")
            if not isinstance(action, int) or action not in range(self.action_size):
                raise ValueError("Invalid action")
            self.memory.append((state, action, reward, next_state, done))
            self.logger.debug({"event": "store_experience", "action": action, "reward": reward})
        except Exception as e:
            self.logger.error({"event": "store_experience_error", "error": str(e)})

    def predict(self, state, balance):
        """Predict high-level strategy and low-level action.

        Args:
            state (list): Current state [price, volume, usd, position].
            balance (dict): Current balance for all pairs.

        Returns:
            int: Action (0: buy, 1: sell, 2: hold).
        """
        try:
            if not (isinstance(state, list) and len(state) == self.state_size):
                raise ValueError(f"State must be a list of length {self.state_size}")
            if not isinstance(balance, dict):
                raise ValueError("Balance must be a dictionary")

            with torch.no_grad():
                meta_state = torch.tensor(state + [balance.get(pair, 0) for pair in TRADING_PAIRS], 
                                        dtype=torch.float32, device=self.device).unsqueeze(0)
                strategy_probs = self.meta_controller(meta_state)
                strategy = torch.argmax(strategy_probs).item()
                
                action_state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
                action_probs = self.sub_policy(action_state)
                action = torch.argmax(action_probs).item()

            self.logger.debug({"event": "hrl_predict", "strategy": strategy, "action": action})
            return action
        except Exception as e:
            self.logger.error({"event": "predict_error", "error": str(e)})
            return 2  # Default to hold on error

    def train(self):
        """Train HRL model using replay buffer and SARSA-like RL with target networks."""
        if len(self.memory) < self.batch_size:
            return

        try:
            batch = random.sample(self.memory, self.batch_size)
            states, actions, rewards, next_states, dones = zip(*batch)

            states = torch.tensor(states, dtype=torch.float32, device=self.device)
            actions = torch.tensor(actions, dtype=torch.long, device=self.device)
            rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
            next_states = torch.tensor(next_states, dtype=torch.float32, device=self.device)
            dones = torch.tensor(dones, dtype=torch.float32, device=self.device)

            meta_states = torch.cat([states, torch.zeros(states.size(0), len(TRADING_PAIRS), device=self.device)], dim=1)
            next_meta_states = torch.cat([next_states, torch.zeros(next_states.size(0), len(TRADING_PAIRS), device=self.device)], dim=1)
            
            meta_q_values = self.meta_controller(meta_states)
            meta_next_q_values = self.meta_target(next_meta_states).detach()
            meta_target_q = rewards + self.gamma * torch.max(meta_next_q_values, dim=1)[0] * (1 - dones)
            meta_q_selected = meta_q_values[range(self.batch_size), actions // self.action_size]
            
            meta_loss = F.mse_loss(meta_q_selected, meta_target_q)
            self.meta_optimizer.zero_grad()
            meta_loss.backward()
            self.meta_optimizer.step()

            sub_q_values = self.sub_policy(states)
            sub_next_q_values = self.sub_target(next_states).detach()
            sub_target_q = rewards + self.gamma * torch.max(sub_next_q_values, dim=1)[0] * (1 - dones)
            sub_q_selected = sub_q_values[range(self.batch_size), actions % self.action_size]
            
            sub_loss = F.mse_loss(sub_q_selected, sub_target_q)
            self.sub_optimizer.zero_grad()
            sub_loss.backward()
            self.sub_optimizer.step()

            for target_param, param in zip(self.meta_target.parameters(), self.meta_controller.parameters()):
                target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
            for target_param, param in zip(self.sub_target.parameters(), self.sub_policy.parameters()):
                target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

            self.logger.info({"event": "hrl_train", "meta_loss": meta_loss.item(), "sub_loss": sub_loss.item()})
        except Exception as e:
            self.logger.error({"event": "hrl_train_error", "error": str(e)})
