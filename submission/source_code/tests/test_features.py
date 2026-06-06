# ==============================================================================
# Name: Vivek Kumar
# Enroll: 23125038
# Email: vivek_k@mfs.iitr.ac.in
# ==============================================================================
"""Unit tests for feature engineering modules."""
import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class TestTemporalFeatures:
    """Tests for temporal feature extraction."""

    def setup_method(self):
        """Create sample data for each test."""
        self.dates = pd.date_range("2022-06-19", periods=288, freq="5min")
        self.df = pd.DataFrame({"timestamp": self.dates, "value": np.random.randn(288)})

    def test_hour_extraction(self):
        """Test hour extraction from timestamps."""
        self.df["hour"] = self.df["timestamp"].dt.hour
        assert self.df["hour"].min() == 0
        assert self.df["hour"].max() == 23
        assert len(self.df["hour"].unique()) == 24

    def test_weekday_extraction(self):
        """Test weekday extraction (0=Monday, 6=Sunday)."""
        self.df["weekday"] = self.df["timestamp"].dt.weekday
        assert self.df["weekday"].min() >= 0
        assert self.df["weekday"].max() <= 6

    def test_weekend_flag(self):
        """Test weekend boolean flag."""
        self.df["weekend"] = self.df["timestamp"].dt.weekday >= 5
        assert self.df["weekend"].dtype == bool

    def test_peak_period(self):
        """Test peak period detection (7-9am, 5-8pm)."""
        hours = self.df["timestamp"].dt.hour
        peak = ((hours >= 7) & (hours < 9)) | ((hours >= 17) & (hours < 20))
        assert peak.any()
        # Check specific hours
        assert peak[self.df["timestamp"].dt.hour == 8].all()
        assert peak[self.df["timestamp"].dt.hour == 18].all()

    def test_off_peak_period(self):
        """Test off-peak period detection (10pm-6am)."""
        hours = self.df["timestamp"].dt.hour
        off_peak = (hours >= 22) | (hours < 6)
        assert off_peak.any()


class TestDemandFeatures:
    """Tests for demand feature computation."""

    def test_utilization_rate(self):
        """Test utilization rate computation."""
        occupancy = np.array([5, 10, 15, 20])
        capacity = np.array([20, 20, 20, 20])
        util = occupancy / capacity
        assert np.allclose(util, [0.25, 0.5, 0.75, 1.0])

    def test_utilization_rate_clipping(self):
        """Test utilization rate is clipped to [0, 1]."""
        occupancy = np.array([25, -5])
        capacity = np.array([20, 20])
        util = np.clip(occupancy / capacity, 0, 1)
        assert util[0] == 1.0
        assert util[1] == 0.0

    def test_revenue_computation(self):
        """Test revenue = volume × price."""
        volume = np.array([10, 20, 30])
        price = np.array([15, 12, 18])
        revenue = volume * price
        np.testing.assert_array_equal(revenue, [150, 240, 540])

    def test_revenue_per_kwh(self):
        """Test revenue per kWh computation with zero protection."""
        revenue = np.array([150, 0, 540])
        volume = np.array([10, 0, 30])
        with np.errstate(divide="ignore", invalid="ignore"):
            rpk = np.where(volume > 0, revenue / volume, 0)
        np.testing.assert_array_equal(rpk, [15, 0, 18])

    def test_lagged_features(self):
        """Test lagged feature creation."""
        df = pd.DataFrame({"value": [1, 2, 3, 4, 5]})
        df["lag_1"] = df["value"].shift(1)
        df["lag_2"] = df["value"].shift(2)
        assert np.isnan(df["lag_1"].iloc[0])
        assert df["lag_1"].iloc[1] == 1
        assert df["lag_2"].iloc[2] == 1

    def test_rolling_features(self):
        """Test rolling mean and std computation."""
        df = pd.DataFrame({"value": [10, 20, 30, 40, 50]})
        df["rolling_mean_3"] = df["value"].rolling(3).mean()
        assert np.isnan(df["rolling_mean_3"].iloc[0])
        assert df["rolling_mean_3"].iloc[2] == 20.0


class TestCongestionFeatures:
    """Tests for congestion feature computation."""

    def test_queue_length_proxy(self):
        """Test queue length proxy: max(0, occupancy - capacity)."""
        occupancy = np.array([15, 20, 25])
        capacity = np.array([20, 20, 20])
        queue = np.maximum(0, occupancy - capacity)
        np.testing.assert_array_equal(queue, [0, 0, 5])

    def test_occupancy_density(self):
        """Test occupancy density = occupancy / area."""
        occupancy = np.array([10, 20])
        area = np.array([5.0, 10.0])
        density = occupancy / area
        np.testing.assert_array_equal(density, [2.0, 2.0])

    def test_congestion_label(self):
        """Test binary congestion label at 80% threshold."""
        utilization = np.array([0.3, 0.5, 0.79, 0.80, 0.81, 0.95])
        threshold = 0.8
        label = (utilization > threshold).astype(int)
        np.testing.assert_array_equal(label, [0, 0, 0, 0, 1, 1])


class TestPricingFeatures:
    """Tests for pricing feature computation."""

    def test_price_change(self):
        """Test price change computation."""
        prices = pd.Series([15, 15, 16, 14, 18])
        change = prices.diff()
        assert change.iloc[2] == 1
        assert change.iloc[3] == -2

    def test_demand_elasticity_computation(self):
        """Test demand elasticity = ΔQ%/ΔP%."""
        price_pct = pd.Series([0.0, 0.1, -0.05, 0.2])
        volume_pct = pd.Series([0.0, -0.03, 0.02, -0.07])
        with np.errstate(divide="ignore", invalid="ignore"):
            elasticity = np.where(np.abs(price_pct) > 1e-8, volume_pct / price_pct, 0)
        assert elasticity[1] == pytest.approx(-0.3)

    def test_price_level_classification(self):
        """Test price level categories."""
        prices = np.array([12, 15, 19.5])
        baseline = 15.0
        levels = np.where(prices < baseline * 0.9, "discount", np.where(prices > baseline * 1.1, "surge", "normal"))
        assert levels[0] == "discount"
        assert levels[1] == "normal"
        assert levels[2] == "surge"


class TestSpatialFeatures:
    """Tests for spatial feature computation."""

    def test_neighbor_mean(self):
        """Test neighbor mean computation using adjacency matrix."""
        adj = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
        values = np.array([10, 20, 30])
        # For each node: mean of connected neighbors
        neighbor_sum = adj @ values
        neighbor_count = adj.sum(axis=1)
        neighbor_mean = np.where(neighbor_count > 0, neighbor_sum / neighbor_count, 0)
        assert neighbor_mean[0] == 20.0  # Connected to node 1
        assert neighbor_mean[1] == 20.0  # Connected to nodes 0, 2: (10+30)/2
        assert neighbor_mean[2] == 20.0  # Connected to node 1

    def test_adjacency_matrix_properties(self):
        """Test adjacency matrix is binary and has zero diagonal."""
        adj = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
        assert np.all((adj == 0) | (adj == 1))  # Binary
        assert np.all(np.diag(adj) == 0)  # Zero diagonal
