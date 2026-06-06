"""Run revenue simulation comparing fixed vs dynamic pricing."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def main():
    logger.info("Starting revenue simulation...")
    try:
        from src.agents.revenue_simulator import RevenueSimulator
        config = {"baseline_price": 15.0}
        simulator = RevenueSimulator(config)
        logger.info("Revenue Simulator initialized. Baseline: ₹15/kWh")
        logger.info("Scenarios: Fixed pricing vs Dynamic (PPO) pricing")
        logger.info("Use: simulator.simulate_fixed_pricing() vs simulator.simulate_dynamic_pricing()")
    except Exception as e:
        logger.error(f"Failed to initialize simulator: {e}")

if __name__ == "__main__":
    main()
