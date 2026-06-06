"""Train PPO tariff pricing agent."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def main():
    logger.info("Starting PPO tariff agent training...")
    try:
        from src.agents.tariff_agent import TariffPricingAgent
        config = {"baseline_price": 15.0, "surge_threshold": 0.8, "discount_threshold": 0.3, "total_timesteps": 100000}
        agent = TariffPricingAgent(config)
        logger.info("PPO Tariff Agent initialized. Baseline: ₹15/kWh")
        logger.info("Use: agent.train(data_df) -> agent.recommend_tariff(obs)")
    except Exception as e:
        logger.error(f"Failed to initialize tariff agent: {e}")

if __name__ == "__main__":
    main()
