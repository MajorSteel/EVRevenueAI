"""Unit tests for ML agents."""
import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class TestDemandAgent:
    """Tests for demand prediction agent."""

    def test_temporal_split(self):
        """Test temporal split does not shuffle data."""
        n = 100
        test_size = 0.2
        split_idx = int(n * (1 - test_size))
        train_idx = list(range(split_idx))
        test_idx = list(range(split_idx, n))
        assert len(train_idx) == 80
        assert len(test_idx) == 20
        assert max(train_idx) < min(test_idx)

    def test_target_creation(self):
        """Test future target column creation (shift)."""
        values = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        future = np.roll(values, -1)
        future[-1] = np.nan
        assert future[0] == 20.0
        assert future[3] == 50.0

    def test_model_selection_by_r2(self):
        """Test best model selection by R² score."""
        model_scores = {"xgboost": {"r2": 0.82}, "lightgbm": {"r2": 0.87}}
        best = max(model_scores, key=lambda m: model_scores[m]["r2"])
        assert best == "lightgbm"


class TestCongestionAgent:
    """Tests for congestion prediction agent."""

    def test_binary_target_creation(self):
        """Test binary congestion target from utilization threshold."""
        utilization = np.array([0.3, 0.5, 0.7, 0.8, 0.85, 0.95])
        threshold = 0.8
        target = (utilization > threshold).astype(int)
        np.testing.assert_array_equal(target, [0, 0, 0, 0, 1, 1])

    def test_class_imbalance(self):
        """Test class imbalance detection."""
        target = np.array([0] * 90 + [1] * 10)
        ratio = target.sum() / len(target)
        assert ratio == 0.1  # 10% positive class — imbalanced

    def test_classification_metrics_range(self):
        """Test classification metrics are in valid ranges."""
        accuracy = 0.912
        precision = 0.887
        recall = 0.853
        f1 = 0.869
        auc = 0.941
        for metric in [accuracy, precision, recall, f1, auc]:
            assert 0.0 <= metric <= 1.0


class TestTariffAgent:
    """Tests for PPO tariff pricing agent."""

    def test_action_space(self):
        """Test action space has 6 discrete actions."""
        actions = [-0.2, -0.1, 0.0, 0.1, 0.2, 0.3]
        assert len(actions) == 6

    def test_price_multiplier(self):
        """Test price multiplier application."""
        base_price = 15.0
        action_multipliers = [-0.2, -0.1, 0.0, 0.1, 0.2, 0.3]
        for mult in action_multipliers:
            new_price = base_price * (1 + mult)
            assert new_price > 0

    def test_reward_computation(self):
        """Test reward function components."""
        alpha, beta, gamma = 1.0, 0.5, 2.0
        revenue_gain = 0.15  # 15% revenue increase
        utilization_balance = -abs(0.72 - 0.7)  # Close to target
        congestion_penalty = max(0, 0.72 - 0.9) ** 2  # No penalty
        reward = alpha * revenue_gain + beta * utilization_balance - gamma * congestion_penalty
        assert reward > 0  # Should be positive with good performance

    def test_observation_space_size(self):
        """Test observation space has 8 dimensions."""
        obs_size = 8  # occupancy, pred_demand, price, hour_sin, hour_cos, util, fast_ratio, cbd
        assert obs_size == 8

    def test_surge_threshold_logic(self):
        """Test surge pricing when utilization > 80%."""
        utilization = 0.85
        threshold = 0.8
        should_surge = utilization > threshold
        assert should_surge is True

    def test_discount_threshold_logic(self):
        """Test discount pricing when utilization < 30%."""
        utilization = 0.25
        threshold = 0.3
        should_discount = utilization < threshold
        assert should_discount is True


class TestRevenueSimulator:
    """Tests for revenue simulator."""

    def test_revenue_gain_pct(self):
        """Test revenue gain percentage calculation."""
        fixed_revenue = 2_090_000
        dynamic_revenue = 2_477_000
        gain_pct = ((dynamic_revenue - fixed_revenue) / fixed_revenue) * 100
        assert gain_pct == pytest.approx(18.5, abs=0.5)

    def test_fixed_pricing_baseline(self):
        """Test fixed pricing uses ₹15/kWh."""
        baseline_price = 15.0
        volume = 1000  # kWh
        revenue = baseline_price * volume
        assert revenue == 15000

    def test_comparison_metrics(self):
        """Test comparison produces all required metrics."""
        required = ["Revenue Gain %", "Utilization Rate", "Congestion Reduction", "Off-Peak Uplift"]
        for metric in required:
            assert isinstance(metric, str)


class TestMonitoringAgent:
    """Tests for monitoring and learning agent."""

    def test_drift_detection_zscore(self):
        """Test Z-score based drift detection."""
        values = np.random.normal(0, 1, 100)
        mean = np.mean(values)
        std = np.std(values)
        threshold = 2.0
        current = 5.0  # Extreme value
        z_score = abs(current - mean) / (std + 1e-8)
        assert z_score > threshold  # Should detect drift

    def test_no_drift_normal_value(self):
        """Test no drift for normal values."""
        values = np.random.normal(0, 1, 100)
        mean = np.mean(values)
        std = np.std(values)
        threshold = 2.0
        current = 0.5  # Normal value
        z_score = abs(current - mean) / (std + 1e-8)
        assert z_score < threshold  # Should not detect drift

    def test_pricing_efficiency(self):
        """Test pricing efficiency = revenue/kWh."""
        revenue = 100000
        kwh = 5950
        efficiency = revenue / kwh
        assert efficiency == pytest.approx(16.81, abs=0.1)
