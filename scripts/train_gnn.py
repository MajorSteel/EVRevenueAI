"""Train GNN spatial demand forecasting model."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def main():
    logger.info("Starting GNN spatial demand forecasting training...")
    try:
        from src.agents.gnn_agent import GNNSpatialAgent
        config = {"hidden_dim": 64, "num_layers": 2, "dropout": 0.1, "epochs": 100, "lr": 0.001}
        agent = GNNSpatialAgent(config)
        logger.info("GNN Agent initialized. Architecture: GCNConv -> GCNConv -> Linear")
        logger.info("Use: agent.prepare_graph_data() -> agent.train() -> agent.evaluate()")
    except Exception as e:
        logger.error(f"Failed to initialize GNN agent: {e}")

if __name__ == "__main__":
    main()
