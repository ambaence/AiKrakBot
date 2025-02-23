# AiKrakBot

AiKrakBot is an advanced trading bot for the Kraken cryptocurrency exchange, featuring a Kivy-based UI, a sophisticated ensemble of machine learning models, and a diverse set of trading strategies with dynamic switching based on market conditions. Built with Python, it leverages cutting-edge AI and risk management techniques to optimize trading performance.

## Features

- **Real-Time Trading**: Streams market data via WebSocket and executes trades using the KrakenAPI with simulation and live modes.
- **Machine Learning Ensemble**: Combines a variety of models for robust price predictions:
  - **LSTM**: Long Short-Term Memory for time-series forecasting.
  - **GRU**: Gated Recurrent Unit for efficient sequence processing.
  - **Transformer**: Multi-head attention for multi-scale dependencies.
  - **TCN**: Temporal Convolutional Network for high-frequency data.
  - **PPO**: Proximal Policy Optimization for reinforcement learning.
  - **Actor-Critic**: Hybrid RL model for trading decisions.
  - **DQN**: Deep Q-Network for Q-learning-based trading.
  - **GNN**: Graph Neural Network for market structure analysis.
  - **Random Forest**: Feature-rich model with SHAP explainability.
  - **GAN**: Wasserstein GAN with Gradient Penalty for synthetic data generation.
  - **HRL**: Hierarchical Reinforcement Learning for strategic decisions.
  - **Sentiment Analysis**: VADER-based sentiment from NewsAPI crypto news.
  - **Autoencoder**: Anomaly detection in market data.
  - **Transfer Learning**: Pre-trained LSTM fine-tuned for crypto markets.
- **Trading Strategies**: Supports a comprehensive set with dynamic selection:
  - **Momentum**: Trades based on price momentum trends.
  - **Scalping**: High-frequency trades with tight thresholds.
  - **Arbitrage**: Triangular arbitrage across BTC/USD, ETH/USD, ETH/BTC.
  - **Mean Reversion**: Trades deviations from historical averages.
  - **Pair Trading**: Exploits BTC/USD and ETH/USD price correlations.
  - **Breakout**: Trades breakouts above resistance or below support.
  - **Grid Trading**: Places buy/sell grids for price oscillation profits.
  - **DCA**: Dollar-Cost Averaging for periodic fixed investments.
- **Risk Management**: Implements stop-loss, take-profit, risk parity, portfolio risk monitoring, and stress testing.
- **Backtesting**: Evaluates strategies using historical or GAN-generated synthetic data.
- **Security**: Encrypts API keys with Fernet, rotates keys periodically, and uses JWT-based authentication with SMS/email 2FA.
- **UI**: Kivy-based dashboard with live price plots, predictions, risk metrics, and strategy controls.

## Installation

1. **Clone the Repository**:
   ```bash
   git clone <repository-url>
   cd AiKrakBot