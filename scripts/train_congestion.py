"""Train congestion prediction models."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agents.congestion_agent import CongestionPredictionAgent
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def main():
    logger.info("Starting congestion prediction training...")
    config = {"threshold": 0.8}
    agent = CongestionPredictionAgent(config)
    logger.info("Congestion agent initialized. Threshold: 80% utilization.")
    logger.info("Use: agent.prepare_data(df) -> agent.train() -> agent.evaluate()")

if __name__ == "__main__":
    main()
