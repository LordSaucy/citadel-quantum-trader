#!/usr/bin/env python3
"""
tests/test_correlation_matrix.py

Tests for correlation matrix computation and asset correlation tracking.

✅ FIXED: 
  1. Modernized numpy.random.Generator (replaced legacy np.random.randn)
  2. Corrected pytest.approx() idiom (expected inside, actual outside)
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
    """
    Generate sample price data for multiple assets.
    
    ✅ FIXED: Use modern numpy.random.Generator instead of legacy np.random.randn()
    """
    dates = pd.date_range("2024-01-01", periods=100, freq="D")
    
    # ✅ FIXED: Create modern RNG instance
    rng = np.random.default_rng(seed=42)
    
    data = pd.DataFrame({
        # ✅ FIXED: Replace legacy np.random.randn() with rng.standard_normal()
        "EURUSD": rng.standard_normal(100).cumsum() + 1.0800,
        "GBPUSD": rng.standard_normal(100).cumsum() + 1.2700,
        "USDJPY": rng.standard_normal(100).cumsum() + 110.0,
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
        # ✅ FIXED: Correct pytest.approx() idiom (expected inside, actual outside)
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
    """
    Test retrieval of current correlation estimates.
    
    ✅ FIXED: Modernized numpy.random.Generator
    """
    dates = pd.date_range("2024-01-01", periods=50, freq="D")
    
    # ✅ FIXED: Create modern RNG instance
    rng = np.random.default_rng(seed=123)
    
    prices = pd.DataFrame({
        # ✅ FIXED: Replace legacy np.random.randn() with rng.standard_normal()
        "EURUSD": rng.standard_normal(50).cumsum() + 1.0800,
        "GBPUSD": rng.standard_normal(50).cumsum() + 1.2700,
    }, index=dates)
    
    # Get correlations
    corrs = get_current_correlations(prices)
    
    # Should have correlations for the two pairs
    assert "EURUSD" in corrs
    assert "GBPUSD" in corrs
    
    # Self-correlation should be 1.0
    # ✅ FIXED: Correct pytest.approx() idiom (expected inside, actual outside)
    assert corrs["EURUSD"]["EURUSD"] == pytest.approx(1.0, rel=1e-8)
    assert corrs["GBPUSD"]["GBPUSD"] == pytest.approx(1.0, rel=1e-8)


def test_correlation_eurusd_gbpusd_positive():
    """
    Test that EURUSD and GBPUSD have positive correlation (typical behavior).
    
    ✅ FIXED: Modernized numpy.random.Generator for reproducible test data
    """
    dates = pd.date_range("2024-01-01", periods=100, freq="D")
    
    # ✅ FIXED: Create modern RNG instance
    rng = np.random.default_rng(seed=999)
    
    # Create correlated data (both respond similarly to market shocks)
    shock = rng.standard_normal(100)
    eu_noise = rng.standard_normal(100) * 0.2
    gb_noise = rng.standard_normal(100) * 0.2
    
    prices = pd.DataFrame({
        "EURUSD": (shock + eu_noise).cumsum() + 1.0800,
        "GBPUSD": (shock + gb_noise).cumsum() + 1.2700,
    }, index=dates)
    
    corr_matrix = compute_correlation_matrix(prices)
    eurusd_gbpusd_corr = corr_matrix.loc["EURUSD", "GBPUSD"]
    
    # Correlation should be significantly positive (> 0.5)
    assert eurusd_gbpusd_corr > 0.5


def test_correlation_stability():
    """
    Test that correlation calculations are stable across different runs
    with the same seed.
    
    ✅ FIXED: Modernized numpy.random.Generator for reproducibility
    """
    dates = pd.date_range("2024-01-01", periods=100, freq="D")
    
    # Run 1: Generate data with seed 42
    rng1 = np.random.default_rng(seed=42)
    prices1 = pd.DataFrame({
        "EURUSD": rng1.standard_normal(100).cumsum() + 1.0800,
        "GBPUSD": rng1.standard_normal(100).cumsum() + 1.2700,
    }, index=dates)
    corr1 = compute_correlation_matrix(prices1)
    
    # Run 2: Generate data with same seed 42
    rng2 = np.random.default_rng(seed=42)
    prices2 = pd.DataFrame({
        "EURUSD": rng2.standard_normal(100).cumsum() + 1.0800,
        "GBPUSD": rng2.standard_normal(100).cumsum() + 1.2700,
    }, index=dates)
    corr2 = compute_correlation_matrix(prices2)
    
    # Correlations should be identical (reproducible)
    assert np.allclose(corr1.values, corr2.values, rtol=1e-10)
