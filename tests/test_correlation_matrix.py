#!/usr/bin/env python3
"""
tests/test_correlation_matrix.py

Tests for correlation matrix computation and asset correlation tracking.

✅ FIXED: Corrected pytest.approx() idiom (expected on inside, not actual)
"""

import numpy as np
import pandas as pd
import pytest

from src.analytics.correlation_matrix import (
    compute_correlation_matrix,
    get_current_correlations,
)


@pytest.fixture
def sample_price_data():
    """Generate sample price data for multiple assets."""
    dates = pd.date_range("2024-01-01", periods=100, freq="D")
    np.random.seed(42)
    
    data = pd.DataFrame({
        "EURUSD": np.random.randn(100).cumsum() + 1.0800,
        "GBPUSD": np.random.randn(100).cumsum() + 1.2700,
        "USDJPY": np.random.randn(100).cumsum() + 110.0,
    }, index=dates)
    
    return data


def test_correlation_matrix_shape(sample_price_data):
    """Test that correlation matrix has the correct shape."""
    corr_matrix = compute_correlation_matrix(sample_price_data)
    
    assert corr_matrix.shape == (3, 3)
    assert all(symbol in corr_matrix.columns for symbol in ["EURUSD", "GBPUSD", "USDJPY"])


def test_correlation_diagonal_is_one(sample_price_data):
    """Test that diagonal elements are 1.0 (perfect correlation with self)."""
    corr_matrix = compute_correlation_matrix(sample_price_data)
    
    # Diagonal should be exactly 1.0
    diagonal = np.diag(corr_matrix.values)
    for diag_val in diagonal:
        # ✅ FIXED: Put expected value inside pytest.approx()
        # Old: assert pytest.approx(diag_val, rel=1e-10) == 1.0
        # New: assert diag_val == pytest.approx(1.0, rel=1e-10)
        assert diag_val == pytest.approx(1.0, rel=1e-10)


def test_correlation_matrix_is_symmetric(sample_price_data):
    """Test that correlation matrix is symmetric."""
    corr_matrix = compute_correlation_matrix(sample_price_data)
    
    # Correlation matrix should be symmetric
    assert np.allclose(corr_matrix.values, corr_matrix.values.T)


def test_correlation_values_in_range(sample_price_data):
    """Test that all correlation values are in [-1, 1]."""
    corr_matrix = compute_correlation_matrix(sample_price_data)
    
    assert (corr_matrix.values >= -1.0).all()
    assert (corr_matrix.values <= 1.0).all()


def test_get_current_correlations():
    """Test retrieval of current correlation estimates."""
    # Generate some test price data
    dates = pd.date_range("2024-01-01", periods=50, freq="D")
    np.random.seed(123)
    
    prices = pd.DataFrame({
        "EURUSD": np.random.randn(50).cumsum() + 1.0800,
        "GBPUSD": np.random.randn(50).cumsum() + 1.2700,
    }, index=dates)
    
    # Get correlations
    corrs = get_current_correlations(prices)
    
    # Should have correlations for the two pairs
    assert "EURUSD" in corrs
    assert "GBPUSD" in corrs
    
    # Self-correlation should be 1.0
    # ✅ FIXED: Put expected value inside pytest.approx()
    # Old: assert pytest.approx(corrs["EURUSD"]["EURUSD"], rel=1e-8) == 1.0
    # New: assert corrs["EURUSD"]["EURUSD"] == pytest.approx(1.0, rel=1e-8)
    assert corrs["EURUSD"]["EURUSD"] == pytest.approx(1.0, rel=1e-8)
    assert corrs["GBPUSD"]["GBPUSD"] == pytest.approx(1.0, rel=1e-8)
