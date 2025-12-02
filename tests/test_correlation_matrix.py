import pandas as pd
import numpy as np
from src.correlation_matrix import rolling_corr_matrix, average_correlation

def test_symmetry_and_diag():
    # Create a dummy DataFrame with 5 assets, 100 rows
    rng = np.random.default_rng(42)
    data = rng.normal(size=(100, 5))
    df = pd.DataFrame(data, columns=[f"S{i}" for i in range(5)])
    corr = rolling_corr_matrix(df)

    # Diagonal must be exactly 1.0
    assert np.allclose(np.diag(corr), np.ones(5))

    # Matrix must be symmetric
    assert np.allclose(corr, corr.T)

def test_average_corr():
    # Perfectly correlated data → avg_corr = 1.0
    df = pd.DataFrame({"A": np.arange(100), "B": np.arange(100) * 2})
    corr = rolling_corr_matrix(df)
    avg = average_correlation(corr)
    assert pytest.approx(avg, rel=1e-6) == 1.0
