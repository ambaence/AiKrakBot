from .lstm_model import LSTMModel
from .dqn_model import DQNModel
from .gru_model import GRUModel
from .transformer_model import TransformerModel
from .tcn_model import TCNModel
from .ppo_model import PPOModel
from .actor_critic import ActorCriticModel
from .autoencoder import AutoencoderModel
from .gnn_model import GNNModel
from .transfer_learning import TransferLearningModel
from .random_forest import RandomForestModel
from .gan_model import GANModel
from .sentiment import SentimentAnalyzer
from .hrl_model import HRLModel
from bayes_opt import BayesianOptimization
from config import GAN_BO_LR_BOUNDS, GAN_BO_BETA1_BOUNDS, GAN_BO_LATENT_DIM_BOUNDS, GAN_BO_MAX_ITER, GAN_PRETRAIN_EPOCHS, LOOKBACK_PERIOD, TRADING_PAIRS
from joblib import Parallel, delayed
import numpy as np
import logging
import os
from collections import deque

# Structured JSON logging
logging.basicConfig(
    filename='logs/bot.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('logs/bot.log', encoding='utf-8')]
)
logger = logging.getLogger(__name__)

class EnsembleModel:
    def __init__(self, api):
        """Initialize ensemble with expanded ML models including enhanced Random Forest and HRL.

        Args:
            api (KrakenAPI): Instance of KrakenAPI for live fee fetching and trade execution.
        """
        self.api = api
        self.lstm = LSTMModel(lookback=20)
        self.dqn = DQNModel(state_size=4, action_size=3)
        self.gru = GRUModel(lookback=20)
        self.transformer = TransformerModel(lookback=20)
        self.tcn = TCNModel(lookback=20)
        self.ppo = PPOModel(state_size=4, action_size=3)
        self.actor_critic = ActorCriticModel(state_size=4, action_size=3)
        self.autoencoder = AutoencoderModel()
        self.gnn = GNNModel()
        self.transfer_learning = TransferLearningModel()
        self.random_forest = RandomForestModel()
        self.gan = GANModel()
        self.sentiment = SentimentAnalyzer()
        self.hrl = HRLModel(state_size=4, action_size=3)
        self.models = [self.lstm, self.gru, self.transformer, self.tcn, self.ppo, self.actor_critic, self.gnn, self.transfer_learning, self.random_forest, self.hrl]
        self.data_buffer = {pair: [] for pair in TRADING_PAIRS}
        self.gan_metrics = {'fid': [], 'mse_moments': [], 'acf_error': [], 'stability': []}
        self.best_gan_params = None
        self.logger = logging.getLogger(__name__)

        self.last_hrl_state = {pair: None for pair in TRADING_PAIRS}
        self.last_hrl_action = {pair: None for pair in TRADING_PAIRS}
        self.last_hrl_balance = {pair: None for pair in TRADING_PAIRS}
        self.last_trade = {pair: None for pair in TRADING_PAIRS}
        self.trade_history = {pair: deque(maxlen=20) for pair in TRADING_PAIRS}
        self.price_history = {pair: deque(maxlen=20) for pair in TRADING_PAIRS}

    def update_data(self, price, volume, balance, position, pair):
        """Update all models with real-time data for a specific pair and store HRL experience.

        Args:
            price (float): Current price.
            volume (float): Current volume.
            balance (dict): Current account balance.
            position (float): Current position for the pair.
            pair (str): Trading pair (e.g., 'BTC/USD').
        """
        try:
            usd = balance.get('USD', 0)
            state = [price, volume, usd, position]
            self.price_history[pair].append(price)
            self.api.increment_fee_tick(pair)

            for model in self.models[:-1]:  # Exclude HRL
                model.update_data(price, volume)

            hrl_action = self.hrl.predict(state, balance)
            if self.last_hrl_state[pair] is not None and self.last_hrl_action[pair] is not None:
                reward = self.calculate_reward(self.last_hrl_state[pair], self.last_hrl_action[pair], state, pair, self.last_hrl_balance[pair], balance)
                self.hrl.store_experience(self.last_hrl_state[pair], self.last_hrl_action[pair], reward, state, False)
                self.logger.debug({"event": "hrl_experience", "pair": pair, "state": self.last_hrl_state[pair], "action": self.last_hrl_action[pair], "reward": reward})

            self.last_hrl_state[pair] = state
            self.last_hrl_action[pair] = hrl_action
            self.last_hrl_balance[pair] = balance.copy()

            self.data_buffer[pair].append([price, volume])
            if len(self.data_buffer[pair]) > 1000:
                self.data_buffer[pair].pop(0)

        except Exception as e:
            self.logger.error({"event": "update_data_error", "pair": pair, "error": str(e)})

    def calculate_reward(self, state, action, next_state, pair, prev_balance, curr_balance):
        """Calculate advanced reward using real profit, portfolio metrics, and live cached fees.

        Args:
            state (list): Previous state [price, volume, usd, position].
            action (int): Action taken (0: buy, 1: sell, 2: hold).
            next_state (list): Next state [price, volume, usd, position].
            pair (str): Trading pair (e.g., 'BTC/USD').
            prev_balance (dict): Balance before action.
            curr_balance (dict): Balance after action.

        Returns:
            float: Reward combining profit, Sortino, Calmar, and live fee-adjusted costs.
        """
        try:
            prev_price, prev_volume, prev_usd, prev_position = state
            next_price, next_volume, next_usd, next_position = next_state

            fee_data = self.api.fetch_trading_fees(pair)
            maker_fee = fee_data.get('maker', 0.0016)

            prices = list(self.price_history[pair])
            if len(prices) >= 2:
                volatility = np.std(prices) / np.mean(prices) if np.mean(prices) != 0 else 0.1
                trade_size = 0.001 / (1 + volatility)
            else:
                trade_size = 0.001

            base, quote = pair.split('/')
            if action == 0:  # Buy
                if prev_usd >= trade_size * prev_price:
                    cost = trade_size * prev_price
                    fee = cost * maker_fee
                    profit = (curr_balance.get('USD', 0) - prev_balance.get('USD', 0)) + \
                             (curr_balance.get(base, 0) - prev_balance.get(base, 0)) * next_price - fee
                else:
                    return -0.5
            elif action == 1:  # Sell
                if prev_position >= trade_size:
                    revenue = trade_size * prev_price
                    fee = revenue * maker_fee
                    profit = (curr_balance.get('USD', 0) - prev_balance.get('USD', 0)) - \
                             (prev_balance.get(base, 0) - curr_balance.get(base, 0)) * prev_price - fee
                else:
                    return -0.5
            else:  # Hold
                profit = 0.0

            if len(prices) >= 2:
                returns = np.diff(prices) / prices[:-1]
                mean_return = np.mean(returns)
                downside_returns = [r for r in returns if r < 0]
                downside_std = np.std(downside_returns) if len(downside_returns) > 0 else 0.1
                sortino = mean_return / downside_std if downside_std != 0 else 0.0

                peak = np.max(prices)
                trough = np.min(prices)
                mdd = (peak - trough) / peak if peak != 0 else 0.0
                annualized_return = mean_return * 365
                calmar = annualized_return / mdd if mdd != 0 else 0.0

                volatility = np.std(prices) / np.mean(prices) if np.mean(prices) != 0 else 0.1
            else:
                sortino = 0.0
                calmar = 0.0
                volatility = 0.1

            transaction_cost = maker_fee * trade_size * prev_price * (1 + volatility)

            reward = profit / 1000.0
            reward += sortino * 0.15
            reward += calmar * 0.1
            reward -= mdd * 0.4
            reward -= transaction_cost * 0.2

            reward = np.clip(reward, -1.0, 1.0)
            return reward

        except Exception as e:
            self.logger.error({"event": "calculate_reward_error", "pair": pair, "error": str(e)})
            return 0.0  # Fallback reward

    def train(self):
        """Train all models in parallel with recovery logic."""
        def train_model_safe(model):
            try:
                model.train()
                self.logger.info({"event": "train_model", "model": model.__class__.__name__, "status": "success"})
            except Exception as e:
                self.logger.error({"event": "train_model_error", "model": model.__class__.__name__, "error": str(e)})
                return None  # Skip failed model

        Parallel(n_jobs=os.cpu_count(), backend="threading")(delayed(train_model_safe)(model) for model in self.models)

        for pair in TRADING_PAIRS:
            if len(self.data_buffer[pair]) >= 100:
                try:
                    gan_data = [self.data_buffer[pair][i:i+LOOKBACK_PERIOD] for i in range(len(self.data_buffer[pair]) - LOOKBACK_PERIOD)]
                    for metrics in self.gan.train(gan_data, batch_size=32):
                        self.gan_metrics['fid'].append(metrics['fid'])
                        self.gan_metrics['mse_moments'].append(metrics['mse_moments'])
                        self.gan_metrics['acf_error'].append(metrics['acf_error'])
                        self.gan_metrics['stability'].append(metrics['stability'])
                        self.logger.info({"event": "gan_train_epoch", "pair": pair, "epoch": metrics['epoch'], 
                                          "fid": metrics['fid'], "mse_moments": metrics['mse_moments'], 
                                          "acf_error": metrics['acf_error'], "stability": metrics['stability']})
                except Exception as e:
                    self.logger.error({"event": "gan_train_error", "pair": pair, "error": str(e)})

    def predict(self, balance, position):
        """Generate predictions with Actor-Critic as primary RL, enhanced RF with SHAP, and HRL.

        Args:
            balance (dict): Current account balance.
            position (float): Current position (assumed for primary pair).

        Returns:
            dict: Prediction with price, action, and individual model outputs.
        """
        try:
            pair = TRADING_PAIRS[0]  # Simplified for primary pair; extend for multi-pair
            usd = balance.get('USD', 0)
            state = [self.lstm.data[-1][0], self.lstm.data[-1][1], usd, position]
            lstm_pred = self.lstm.predict()
            gru_pred = self.gru.predict()
            trans_pred = self.transformer.predict()
            tcn_pred = self.tcn.predict()
            ppo_pred = self.ppo.get_price_prediction(state)
            ac_action = self.actor_critic.predict(state)
            ac_pred = self.actor_critic.get_price_prediction(state)
            gnn_pred = self.gnn.predict()
            tl_pred = self.transfer_learning.predict()
            rf_pred = self.random_forest.predict()
            hrl_action = self.hrl.predict(state, balance)
            anomaly = self.autoencoder.detect_anomaly(state)

            rf_shap = self.random_forest.get_shap_values() if self.random_forest.is_trained else None

            sentiment_score = self.sentiment.get_sentiment()
            preds = [lstm_pred, gru_pred, trans_pred, tcn_pred, ppo_pred, ac_pred, gnn_pred, tl_pred, rf_pred]
            weights = [0.15, 0.15, 0.15, 0.10, 0.10, 0.10, 0.05, 0.10, 0.15]
            if rf_shap is not None and rf_shap.size > 0:
                shap_impact = np.mean(np.abs(rf_shap))
                rf_weight = min(0.15 + shap_impact * 0.01, 0.3)
                weights[8] = rf_weight
                self.logger.info({"event": "rf_weight_adjust", "shap_impact": shap_impact, "new_weight": rf_weight})

            valid_preds = [p for p in preds if p is not None]
            valid_weights = [w for p, w in zip(preds, weights) if p is not None]
            price_pred = sum(p * w for p, w in zip(valid_preds, valid_weights)) / sum(valid_weights) if valid_preds else None
            
            if price_pred and not anomaly:
                price_pred *= (1 + sentiment_score * 0.1)
            
            final_action = hrl_action if hrl_action is not None else ac_action if not anomaly else 2

            return {
                'price': price_pred,
                'action': final_action,
                'individual_preds': {
                    'LSTM': lstm_pred,
                    'GRU': gru_pred,
                    'Transformer': trans_pred,
                    'TCN': tcn_pred,
                    'PPO': ppo_pred,
                    'ActorCritic': ac_pred,
                    'GNN': gnn_pred,
                    'TransferLearning': tl_pred,
                    'RandomForest': rf_pred,
                    'HRL': price_pred
                },
                'sentiment_score': sentiment_score,
                'anomaly': anomaly
            }
        except Exception as e:
            self.logger.error({"event": "predict_error", "pair": pair, "error": str(e)})
            return {'price': None, 'action': 2, 'individual_preds': {}, 'sentiment_score': 0.0, 'anomaly': True}

    def generate_synthetic_batch(self, num_samples=32):
        """Generate synthetic data using tuned GAN."""
        try:
            return self.gan.generate_synthetic_data(num_samples)
        except Exception as e:
            self.logger.error({"event": "generate_synthetic_error", "error": str(e)})
            return []

    def get_gan_metrics(self):
        """Return latest GAN tuning metrics."""
        try:
            return {
                'fid': self.gan_metrics['fid'][-1] if self.gan_metrics['fid'] else 0.0,
                'mse_moments': self.gan_metrics['mse_moments'][-1] if self.gan_metrics['mse_moments'] else 0.0,
                'acf_error': self.gan_metrics['acf_error'][-1] if self.gan_metrics['acf_error'] else 0.0,
                'stability': self.gan_metrics['stability'][-1] if self.gan_metrics['stability'] else 0.0
            }
        except Exception as e:
            self.logger.error({"event": "get_gan_metrics_error", "error": str(e)})
            return {'fid': 0.0, 'mse_moments': 0.0, 'acf_error': 0.0, 'stability': 0.0}

    def tune_gan(self, real_data):
        """Tune GAN hyperparameters using Bayesian optimization."""
        def gan_objective(learning_rate, beta1, latent_dim):
            try:
                gan = GANModel(learning_rate=learning_rate, beta1=beta1, latent_dim=latent_dim)
                gan_data = np.array(real_data[:100])
                for metrics in gan.train(gan_data, epochs=10, batch_size=32):
                    fid = metrics['fid']
                return -fid
            except Exception as e:
                self.logger.error({"event": "gan_objective_error", "error": str(e)})
                return float('-inf')

        try:
            pbounds = {
                'learning_rate': GAN_BO_LR_BOUNDS,
                'beta1': GAN_BO_BETA1_BOUNDS,
                'latent_dim': GAN_BO_LATENT_DIM_BOUNDS
            }
            optimizer = BayesianOptimization(f=gan_objective, pbounds=pbounds, random_state=1, verbose=2)
            optimizer.maximize(init_points=2, n_iter=GAN_BO_MAX_ITER)

            best_params = optimizer.max['params']
            best_params['latent_dim'] = int(best_params['latent_dim'])
            self.best_gan_params = best_params
            self.logger.info({"event": "gan_tune_complete", "best_params": best_params, "fid": -optimizer.max['target']})

            self.gan = GANModel(learning_rate=best_params['learning_rate'], beta1=best_params['beta1'], latent_dim=best_params['latent_dim'])
            for _ in self.gan.train(real_data, epochs=GAN_PRETRAIN_EPOCHS):
                pass
        except Exception as e:
            self.logger.error({"event": "tune_gan_error", "error": str(e)})