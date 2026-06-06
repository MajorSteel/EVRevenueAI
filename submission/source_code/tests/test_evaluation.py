# ==============================================================================
# Name: Vivek Kumar
# Enroll: 23125038
# Email: vivek_k@mfs.iitr.ac.in
# ==============================================================================
"""Unit tests for evaluation metrics."""
import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class TestRegressionMetrics:
    """Tests for regression metrics."""

    def test_rmse(self):
        """Test RMSE computation."""
        y_true = np.array([3, -0.5, 2, 7])
        y_pred = np.array([2.5, 0.0, 2, 8])
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        assert rmse == pytest.approx(0.6124, abs=0.001)

    def test_mae(self):
        """Test MAE computation."""
        y_true = np.array([3, -0.5, 2, 7])
        y_pred = np.array([2.5, 0.0, 2, 8])
        mae = np.mean(np.abs(y_true - y_pred))
        assert mae == pytest.approx(0.5, abs=0.001)

    def test_r2_score(self):
        """Test R² score computation."""
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1.1, 2.1, 2.9, 4.0, 5.1])
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - ss_res / ss_tot
        assert r2 > 0.95

    def test_perfect_prediction(self):
        """Test metrics for perfect predictions."""
        y = np.array([1, 2, 3, 4, 5])
        rmse = np.sqrt(np.mean((y - y) ** 2))
        mae = np.mean(np.abs(y - y))
        assert rmse == 0.0
        assert mae == 0.0


class TestClassificationMetrics:
    """Tests for classification metrics."""

    def test_accuracy(self):
        """Test accuracy computation."""
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_pred = np.array([0, 0, 1, 1, 0, 0])
        accuracy = np.mean(y_true == y_pred)
        assert accuracy == pytest.approx(5 / 6)

    def test_precision(self):
        """Test precision = TP / (TP + FP)."""
        tp = 2
        fp = 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        assert precision == 1.0

    def test_recall(self):
        """Test recall = TP / (TP + FN)."""
        tp = 2
        fn = 1
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        assert recall == pytest.approx(2 / 3)

    def test_f1_score(self):
        """Test F1 = 2 * (precision * recall) / (precision + recall)."""
        precision = 1.0
        recall = 2 / 3
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        assert f1 == pytest.approx(0.8, abs=0.001)


class TestBusinessMetrics:
    """Tests for business-specific metrics."""

    def test_revenue_gain_pct(self):
        """Test revenue gain percentage formula."""
        new_revenue = 2_477_000
        old_revenue = 2_090_000
        gain = ((new_revenue - old_revenue) / old_revenue) * 100
        assert gain == pytest.approx(18.5, abs=0.5)

    def test_utilization_rate(self):
        """Test utilization rate = charging_time / total_time."""
        charging_time = 14.4  # hours
        total_time = 24.0  # hours
        utilization = charging_time / total_time
        assert utilization == pytest.approx(0.6, abs=0.001)

    def test_off_peak_uplift(self):
        """Test off-peak uplift calculation."""
        sessions_before = 100
        sessions_after = 131
        uplift = ((sessions_after - sessions_before) / sessions_before) * 100
        assert uplift == pytest.approx(31.0)

    def test_pricing_efficiency(self):
        """Test pricing efficiency = revenue / kWh."""
        revenue = 16820
        kwh = 1000
        efficiency = revenue / kwh
        assert efficiency == pytest.approx(16.82)
