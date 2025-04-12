import os
import logging
from backend.api_handler import KrakenAPI
from backend.ml_engine.ensemble import EnsembleModel
from backend.risk_management import RiskManager
from backend.strategies.manager import StrategyManager
from config import TRADING_PAIRS, SIMULATE

# Configure debug-level logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def validate_env_vars():
    """Validate required environment variables."""
    required_vars = [
        "ENCRYPTED_KRAKEN_API_KEY",
        "ENCRYPTED_KRAKEN_API_SECRET",
        "NEWSAPI_KEY"
    ]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        logger.error(f"Missing environment variables: {', '.join(missing_vars)}")
    else:
        logger.info("All required environment variables are set.")

def test_api_connectivity():
    """Test Kraken API connectivity."""
    try:
        api = KrakenAPI(simulate=SIMULATE)
        balance = api.get_balance()
        logger.info(f"API connectivity test passed. Balance: {balance}")
    except Exception as e:
        logger.error(f"API connectivity test failed: {e}")

def test_model_initialization():
    """Test model initialization."""
    try:
        api = KrakenAPI(simulate=SIMULATE)
        model = EnsembleModel(api)
        logger.info("Model initialization test passed.")
    except Exception as e:
        logger.error(f"Model initialization test failed: {e}")

def test_strategy_manager():
    """Test strategy manager initialization."""
    try:
        api = KrakenAPI(simulate=SIMULATE)
        model = EnsembleModel(api)
        risk_manager = RiskManager(api)
        strategy_manager = StrategyManager(api, model, risk_manager)
        logger.info("Strategy manager test passed.")
    except Exception as e:
        logger.error(f"Strategy manager test failed: {e}")

if __name__ == "__main__":
    logger.info("Starting debug tests...")
    validate_env_vars()
    test_api_connectivity()
    test_model_initialization()
    test_strategy_manager()
    logger.info("Debug tests completed.")
