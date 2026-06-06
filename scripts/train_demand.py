"""Train demand prediction models."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agents.demand_agent import DemandPredictionAgent
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def main():
    logger.info("Starting demand prediction training...")
    config = {"target": "future_volume", "test_size": 0.2}
    agent = DemandPredictionAgent(config)
    logger.info("Demand agent initialized. Ready for training with data.")
    logger.info("Use: agent.prepare_data(df) -> agent.train() -> agent.evaluate()")

if __name__ == "__main__":
    main()
