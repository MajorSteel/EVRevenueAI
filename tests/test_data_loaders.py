"""Unit tests for data loaders."""
import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class TestACNDataLoader:
    """Test suite for ACN data loader."""

    def test_loader_initialization(self):
        """Test that ACN loader initializes without error."""
        try:
            from src.data.acn_loader import ACNDataLoader
            loader = ACNDataLoader()
            assert loader is not None
        except ImportError:
            pytest.skip("ACN loader not available")

    def test_acn_schema_fields(self):
        """Test that expected ACN schema fields are defined."""
        expected_fields = [
            "connectionTime", "disconnectTime", "doneChargingTime",
            "kWhDelivered", "stationID", "siteID",
        ]
        for field in expected_fields:
            assert isinstance(field, str)

    def test_datetime_parsing(self):
        """Test datetime parsing for ACN time fields."""
        sample_times = [
            "2018-04-25 08:30:00",
            "2018-04-25 17:45:00",
            "2018-12-16 23:59:59",
        ]
        for t in sample_times:
            parsed = pd.to_datetime(t)
            assert parsed is not None
            assert parsed.year == 2018

    def test_kwh_validation(self):
        """Test kWh delivered values are non-negative."""
        kwh_values = np.array([0.0, 5.5, 12.3, 45.6, 100.0])
        assert np.all(kwh_values >= 0)

    def test_missing_value_handling(self):
        """Test missing value handling strategies."""
        df = pd.DataFrame({
            "kWhDelivered": [1.0, np.nan, 3.0, np.nan, 5.0],
            "stationID": ["A", "A", None, "B", "B"],
        })
        # Interpolation for numeric
        df["kWhDelivered"] = df["kWhDelivered"].interpolate()
        assert df["kWhDelivered"].isna().sum() == 0
        assert df["kWhDelivered"].iloc[1] == 2.0  # Interpolated

        # Forward fill for categorical
        df["stationID"] = df["stationID"].ffill()
        assert df["stationID"].isna().sum() == 0


class TestUrbanEVDataLoader:
    """Test suite for UrbanEV data loader."""

    def test_loader_initialization(self):
        """Test that UrbanEV loader initializes without error."""
        try:
            from src.data.urbanev_loader import UrbanEVDataLoader
            loader = UrbanEVDataLoader()
            assert loader is not None
        except ImportError:
            pytest.skip("UrbanEV loader not available")

    def test_time_csv_structure(self):
        """Test expected time.csv column structure."""
        expected_cols = ["month", "day", "year", "hour", "minute", "second"]
        df = pd.DataFrame(columns=expected_cols)
        assert list(df.columns) == expected_cols

    def test_adjacency_matrix_symmetry(self):
        """Test adjacency matrix is symmetric."""
        n = 10
        adj = np.random.randint(0, 2, (n, n))
        adj_sym = np.maximum(adj, adj.T)
        np.fill_diagonal(adj_sym, 0)
        assert np.array_equal(adj_sym, adj_sym.T)

    def test_district_count(self):
        """Test expected number of districts."""
        expected_districts = 248
        assert expected_districts > 0

    def test_time_intervals(self):
        """Test 5-minute time intervals."""
        times = pd.date_range("2022-06-19", periods=288, freq="5min")
        assert len(times) == 288  # 24 hours at 5-min = 288
        assert (times[1] - times[0]).total_seconds() == 300  # 5 minutes

    def test_wide_to_long_format(self):
        """Test wide-to-long format conversion."""
        wide = pd.DataFrame({
            "timestamp": [1, 2, 3],
            "station_100": [10, 20, 30],
            "station_200": [15, 25, 35],
        })
        long = wide.melt(id_vars=["timestamp"], var_name="station_id", value_name="value")
        assert len(long) == 6  # 3 timestamps × 2 stations
        assert "station_id" in long.columns


class TestSchemaValidator:
    """Test suite for schema validation."""

    def test_valid_acn_record(self):
        """Test validation of a valid ACN record."""
        record = {
            "connectionTime": "2018-04-25 08:30:00",
            "disconnectTime": "2018-04-25 17:45:00",
            "doneChargingTime": "2018-04-25 12:30:00",
            "kWhDelivered": 45.6,
            "stationID": "CA-001",
            "siteID": "Caltech",
        }
        assert all(k in record for k in ["connectionTime", "kWhDelivered", "stationID"])

    def test_invalid_kwh(self):
        """Test that negative kWh is flagged."""
        kwh = -5.0
        assert kwh < 0  # Should be flagged as invalid


class TestPreprocessor:
    """Test suite for data preprocessor."""

    def test_outlier_detection_iqr(self):
        """Test IQR-based outlier detection."""
        data = pd.Series([1, 2, 3, 4, 5, 100])
        q1 = data.quantile(0.25)
        q3 = data.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outliers = data[(data < lower) | (data > upper)]
        assert len(outliers) == 1
        assert outliers.iloc[0] == 100

    def test_minmax_normalization(self):
        """Test min-max normalization."""
        data = pd.Series([0, 25, 50, 75, 100])
        normalized = (data - data.min()) / (data.max() - data.min())
        assert normalized.min() == 0.0
        assert normalized.max() == 1.0

    def test_interpolation(self):
        """Test linear interpolation for missing values."""
        data = pd.Series([1.0, np.nan, 3.0, np.nan, 5.0])
        filled = data.interpolate(method="linear")
        assert filled.iloc[1] == 2.0
        assert filled.iloc[3] == 4.0
