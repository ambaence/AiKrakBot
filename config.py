import os
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# --- Sensitive/Environment-Specific Settings ---
ENCRYPTED_API_KEY = os.getenv("ENCRYPTED_KRAKEN_API_KEY")
ENCRYPTED_API_SECRET = os.getenv("ENCRYPTED_KRAKEN_API_SECRET")
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")
DEFAULT_USERNAME = os.getenv("DEFAULT_USERNAME", "testuser")
DEFAULT_PASSWORD = os.getenv("DEFAULT_PASSWORD", "password123")
GAN_TUNING_PAIR = os.getenv("GAN_TUNING_PAIR", "BTC/USD")
JWT_SECRET = os.getenv("JWT_SECRET", "your-secret-key-here")
JWT_EXPIRY = 3600  # 1 hour
SMS_2FA_TEST_NUMBER = os.getenv("SMS_2FA_TEST_NUMBER", "1234567890")
EMAIL_2FA_TEST_ADDRESS = os.getenv("EMAIL_2FA_TEST_ADDRESS", "test@example.com")
ENCRYPTION_KEY = os.getenv("ENCRYPTION_KEY")
SIMULATE = os.getenv("SIMULATE", "True").lower() == "true"
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# --- Trading Pairs and Settings ---
TRADING_PAIRS = ["BTC/USD", "ETH/USD", "ETH/BTC"]
MAX_TRADE_SIZE = 0.001
RISK_TOLERANCE = 0.02
MAX_PORTFOLIO_RISK = 0.05
STOP_LOSS_MULTIPLIER = 1.5
LOOKBACK_PERIOD = 20
MAKER_FEE = 0.0016  # 0.16%
TAKER_FEE = 0.0026  # 0.26%
NEWS_CACHE_TTL = 900  # 15 minutes

# --- PPO Hyperparameters ---
PPO_GAMMA = 0.99
PPO_CLIP = 0.2
PPO_LR = 0.0003
PPO_MEMORY_SIZE = 2000
PPO_BATCH_SIZE = 64

# --- Actor-Critic Hyperparameters ---
AC_GAMMA = 0.99
AC_LR_ACTOR = 0.0001
AC_LR_CRITIC = 0.0002
AC_MEMORY_SIZE = 2000
AC_BATCH_SIZE = 64

# --- Autoencoder Hyperparameters ---
AE_LATENT_DIM = 10
AE_LR = 0.001
AE_ANOMALY_THRESHOLD = 0.05

# --- GNN Hyperparameters ---
GNN_LR = 0.001
GNN_HIDDEN_DIM = 64
GNN_LAYERS = 2
GNN_CHANNELS = 16

# --- Transfer Learning Hyperparameters ---
TL_LR = 0.0001
TL_PRETRAINED_LAYERS = 2
TL_FINE_TUNE_EPOCHS = 5

# --- Random Forest Hyperparameters ---
RF_N_ESTIMATORS = 100
RF_MAX_DEPTH = 10
RF_MIN_SAMPLES_SPLIT = 2
RF_RSI_PERIOD = 14
RF_MACD_FAST = 12
RF_MACD_SLOW = 26
RF_MACD_SIGNAL = 9
RF_BB_PERIOD = 20
RF_BB_STD = 2
RF_EMA_SHORT = 5
RF_EMA_LONG = 20
RF_MOMENTUM_PERIOD = 10

# --- GAN Hyperparameters ---
GAN_LATENT_DIM = 128
GAN_PRETRAIN_LR = 0.0001
GAN_FINETUNE_LR = 0.00005
GAN_BETA1 = 0.5
GAN_PRETRAIN_EPOCHS = 50
GAN_BATCH_SIZE = 64
GAN_GP_LAMBDA = 10
GAN_CRITIC_ITERS = 5
GAN_FID_THRESHOLD = 30.0
GAN_MSE_THRESHOLD = 0.005
GAN_ACF_THRESHOLD = 0.03
GAN_STABILITY_THRESHOLD = 0.05
GAN_METRIC_UPDATE_FREQ = 5

# --- Bayesian Optimization Bounds ---
GAN_BO_LR_BOUNDS = (1e-5, 1e-3)
GAN_BO_BETA1_BOUNDS = (0.1, 0.9)
GAN_BO_LATENT_DIM_BOUNDS = (64, 256)
GAN_BO_MAX_ITER = 10

# --- Strategy Parameters ---
PAIR_CORR_THRESHOLD = 0.8
BREAKOUT_THRESHOLD = 0.005
GRID_LEVELS = 5
GRID_SPACING = 0.01
DCA_AMOUNT = 10.0
DCA_INTERVAL = 3600  # 1 hour

# --- API Key Rotation Settings ---
KEY_ROTATION_INTERVAL = 86400  # 24 hours
