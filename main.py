import asyncio
import os
import logging
from flask import Flask, jsonify
from backend.api_handler import KrakenAPI
from backend.ml_engine.ensemble import EnsembleModel
from backend.risk_management import RiskManager
from backend.strategies.manager import StrategyManager
from config import TRADING_PAIRS, LOG_LEVEL, SIMULATE  # Import from config

# Structured JSON logging
logging.basicConfig(
    filename='logs/bot.log',
    level=getattr(logging, LOG_LEVEL),
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('logs/bot.log', encoding='utf-8')]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

@app.route("/")
def index():
    return jsonify({"message": "Welcome to AiKrakBot Web UI"})

@app.route("/status")
def status():
    return jsonify({"status": "running"})

async def main():
    try:
        simulate = SIMULATE  # Use from config.py
        required_env_vars = ["ENCRYPTED_KRAKEN_API_KEY", "ENCRYPTED_KRAKEN_API_SECRET"]
        missing_vars = [var for var in required_env_vars if not os.getenv(var)]
        if not simulate and missing_vars:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")

        logger.info({"event": "initialization_start", "simulate": simulate})
        
        api = KrakenAPI(simulate=simulate)
        api.get_balance()  # Validate API key permissions

        model = EnsembleModel(api)
        risk_manager = RiskManager(api)  # Ensure only 'api' is passed
        strategy_manager = StrategyManager(api, model, risk_manager)
        
        for pair in TRADING_PAIRS:
            try:
                api.fetch_trading_fees(pair)
                logger.info({"event": "fee_prefetch", "pair": pair, "status": "success"})
            except Exception as e:
                logger.error({"event": "fee_prefetch_failed", "pair": pair, "error": str(e)})

        logger.info({"event": "initialization_complete"})
        app.run(host="0.0.0.0", port=5000)

    except ValueError as e:
        logger.error({"event": "initialization_error", "error": str(e)})
        raise
    except Exception as e:
        logger.error({"event": "unexpected_error", "error": str(e)})
        raise

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info({"event": "shutdown", "reason": "user_interrupt"})
    except Exception as e:
        logger.error({"event": "fatal_error", "error": str(e)})
