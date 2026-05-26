"""
Synthetic data generator for benchmarking Bayesian neural networks.

Generates inputs from a standard normal distribution and targets as a nonlinear
additive function of the first four features, plus Gaussian noise. Supports train-test splitting.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons
from ucimlrepo import fetch_ucirepo
from numpy.linalg import eigh
from scipy.stats import norm

def nearest_correlation_matrix(A, eps=1e-8, max_iter=5):
    """
    Higham-style projection:
    1) symmetrize
    2) eigen-decompose and clip negative eigenvalues
    3) renormalize to unit diagonal (correlation)
    Repeat a few times for stability.
    """
    X = (A + A.T) / 2
    for _ in range(max_iter):
        # Eigen clip
        w, V = eigh(X)
        w_clipped = np.maximum(w, eps)
        X = (V * w_clipped) @ V.T
        # Force exact symmetry
        X = (X + X.T) / 2
        # Scale to correlation
        d = np.sqrt(np.clip(np.diag(X), eps, None))
        Dinv = np.diag(1.0 / d)
        X = Dinv @ X @ Dinv
        X = (X + X.T) / 2
        np.fill_diagonal(X, 1.0)
    return X

def spearman_to_gaussian_corr(S):
    """
    Map a Spearman correlation matrix S to the Gaussian copula
    correlation matrix R via R_ij = 2 sin(pi * S_ij / 6).
    """
    S = np.asarray(S)
    if S.shape[0] != S.shape[1]:
        raise ValueError("S must be square.")
    R = 2.0 * np.sin(np.pi * S / 6.0)
    np.fill_diagonal(R, 1.0)
    return R

def sample_gaussian_copula_uniform(n, S, random_state=None):
    """
    Sample n rows of U ~ Gaussian copula with target Spearman matrix S.
    Returns an (n, d) array with uniform(0,1) marginals.
    """
    rng = np.random.default_rng(random_state)
    d = S.shape[0]
    # Map Spearman -> Gaussian copula correlation
    R0 = spearman_to_gaussian_corr(S)
    # Project to nearest valid correlation matrix
    R = nearest_correlation_matrix(R0)
    # Cholesky (add tiny jitter if needed)
    jitter = 0
    for _ in range(3):
        try:
            L = np.linalg.cholesky(R + jitter * np.eye(d))
            break
        except np.linalg.LinAlgError:
            jitter = 1e-10 if jitter == 0 else jitter * 10
    # Sample MVN(0, R)
    Z = rng.standard_normal(size=(n, d)) @ L.T
    # Push through Phi to get uniforms
    U = norm.cdf(Z)
    return U, R  # returning R lets you inspect the actual copula correlation used

def generate_Friedman_data(N=200, D=10, sigma=1.0, test_size=0.2, seed=42, standardize_y=True):
    """
    Generate synthetic regression data for Bayesian neural network experiments.

    Parameters:
        N (int): Number of samples.
        D (int): Number of features.
        sigma (float): Noise level.
        test_size (float): Proportion for test split.
        seed (int): Random seed.
        standardize_y (bool): Whether to standardize the response variable.

    Returns:
        tuple: (X_train, X_test, y_train, y_test, y_mean, y_std) if standardize_y,
               else (X_train, X_test, y_train, y_test)
    """
    np.random.seed(seed)
    X = np.random.uniform(0, 1, size=(N, D))
    x0, x1, x2, x3, x4 = X[:, 0], X[:, 1], X[:, 2], X[:, 3], X[:, 4]

    y_clean = (
        10 * np.sin(np.pi * x0 * x1) +
        20 * (x2 - 0.5) ** 2 +
        10 * x3 +
        5.0 * x4
    )

    y = y_clean + np.random.normal(0, sigma, size=N)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)

    if standardize_y:
        y_mean = y_train.mean()
        y_std = y_train.std() if y_train.std() > 0 else 1.0  # avoid division by zero

        y_train = (y_train - y_mean) / y_std
        y_test = (y_test - y_mean) / y_std

        return X_train, X_test, y_train, y_test

    return X_train, X_test, y_train, y_test

def generate_correlated_Friedman_data(N=100, D=10, sigma=1.0, test_size=0.2, seed=42, standardize_y=True):
    """
    Generate synthetic regression data for Bayesian neural network experiments.

    Parameters:
        N (int): Number of samples.
        D (int): Number of features.
        sigma (float): Noise level.
        test_size (float): Proportion for test split.
        seed (int): Random seed.
        standardize_y (bool): Whether to standardize the response variable.

    Returns:
        tuple: (X_train, X_test, y_train, y_test, y_mean, y_std) if standardize_y,
               else (X_train, X_test, y_train, y_test)
    """
    np.random.seed(seed)
    d = 10
    S_custom = np.eye(d)
    # Block 1 (vars 0..4): high Spearman, 0.7
    for i in range(0, 3):
        for j in range(i+1, 3):
            S_custom[i, j] = S_custom[j, i] = 0.8
    # Block 2 (vars 5..9): moderate Spearman, 0.4
    for i in range(5, 10):
        for j in range(i+1, 10):
            S_custom[i, j] = S_custom[j, i] = -0.5
    # Cross-block weaker, 0.15
    for i in range(0, 5):
        for j in range(5, 10):
            S_custom[i, j] = S_custom[j, i] = 0.15
    # A couple of bespoke pairs:
    S_custom[0, 9] = S_custom[9, 0] = 0.4
    S_custom[2, 7] = S_custom[7, 2] = 0.9  # very strong (will be projected if infeasible)
    S_custom[3, 4] = S_custom[4, 3] = -0.9  # very strong (will be projected if infeasible)
    S_custom[1, 6] = S_custom[6, 1] = -0.9  # very strong (will be projected if infeasible)

    U, _ = sample_gaussian_copula_uniform(n=10000, S=S_custom, random_state=123)
    #X = np.random.uniform(0, 1, size=(N, D))
    if N != U.shape[0]:
        idx = np.random.choice(U.shape[0], size=N, replace=False)
        X = U[idx, :]
    else:
        X = U

    x0, x1, x2, x3, x4 = X[:, 0], X[:, 1], X[:, 2], X[:, 3], X[:, 4]

    y_clean = (
        10 * np.sin(np.pi * x0 * x1) +
        20 * (x2 - 0.5) ** 2 +
        10 * x3 +
        5.0 * x4
    )

    y = y_clean + np.random.normal(0, sigma, size=N)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)

    if standardize_y:
        y_mean = y_train.mean()
        y_std = y_train.std() if y_train.std() > 0 else 1.0  # avoid division by zero

        y_train = (y_train - y_mean) / y_std
        y_test = (y_test - y_mean) / y_std

        return X_train, X_test, y_train, y_test

    return X_train, X_test, y_train, y_test


def load_abalone_regression_data(
    path="datasets/abalone/abalone.csv",
    target="Rings",
    frac=0.5,
    standardized = False,
    random_state=42,
    test_size=0.2
):
    """
    Load and preprocess the abalone dataset for regression.

    Parameters:
        path (str): Path to the abalone CSV file.
        frac (float): Fraction of data to sample.
        random_state (int): Random seed.
        test_size (float): Proportion of data to use for testing.

    Returns:
        X_train, X_test, y_train, y_test
    """
    column_names = [
        "Sex", "Length", "Diameter", "Height",
        "Whole weight", "Shucked weight", "Viscera weight",
        "Shell weight", "Rings"
    ]

    abalone = pd.read_csv(path, header=None, names=column_names)

    # Map sex to integers: I=1, F=2, M=3
    abalone['Sex'] = abalone['Sex'].map({'I': 1, 'F': 2, 'M': 3})
    abalone = abalone.sample(frac=frac, random_state=random_state).reset_index(drop=True)
    
    X = abalone.drop([target], axis=1)
    numeric_cols = X.select_dtypes(include='number').columns.drop('Sex')
    scaler = StandardScaler()
    X_scaled = X.copy()
    X_scaled[numeric_cols] = scaler.fit_transform(X[numeric_cols])
    y_raw = abalone[target].astype(int)
    y = y_raw.values.astype(float)
    
    if standardized:
        return train_test_split(X_scaled, y, test_size=test_size, random_state=random_state)
    else:
        return train_test_split(X, y, test_size=test_size, random_state=random_state)

   
def load_breast_cancer_data(test_size=0.2, standardize=True, random_state=42):
    """
    Load the UCI Breast Cancer Wisconsin (Diagnostic) dataset, split into train/test, optionally standardize.

    Parameters:
        test_size (float): Proportion of the data to include in the test split.
        standardize (bool): Whether to standardize the feature matrix X.
        random_state (int): Random seed for reproducibility.

    Returns:
        X_train, X_test, y_train, y_test, (mean_, scale_) if standardize else (None, None)
    """
    # Load data from local file or download manually first:
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"
    columns = ['ID', 'Diagnosis'] + [f'feature_{i}' for i in range(1, 31)]
    
    data = pd.read_csv(url, header=None, names=columns)
    
    # Map target labels to binary
    data['Diagnosis'] = data['Diagnosis'].map({'M': 2, 'B': 1})

    X = data.drop(['ID', 'Diagnosis'], axis=1).values
    y = data['Diagnosis'].values

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    mean_, scale_ = None, None
    if standardize:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        mean_, scale_ = scaler.mean_, scaler.scale_

    return X_train, X_test, y_train, y_test, mean_, scale_


def generate_linreg_simple_data(
    N=250,
    p=10,
    rho=0.0,
    sigma=1.0,
    seed=123
):
    """
    Generate a simple linear regression dataset:
    y = X beta + noise,
    with optional AR(1)-type correlation structure (rho).
    No interactions and sparse true coefficients.
    """

    np.random.seed(seed)

    # --- Sparse true coefficients ---
    beta_true = np.array([3.0, -2.0, 1.5, 0.8, 0.2] + [0.0]*(p-5))
    beta_true = beta_true[:p]  # ensure correct dimension

    # --- Covariance structure (Toeplitz/AR-like) ---
    if rho == 0.0:
        # Independent predictors
        X = np.random.normal(0, 1, size=(N, p))
    else:
        # Correlated predictors
        Sigma = rho ** np.abs(np.subtract.outer(np.arange(p), np.arange(p)))
        L = np.linalg.cholesky(Sigma)
        X = np.random.normal(size=(N, p)) @ L.T

    # --- Generate y ---
    noise = np.random.normal(0.0, sigma, size=N)
    y = X @ beta_true + noise

    return X, y, beta_true

def load_linreg_dataset(
    path="datasets/linreg/linreg_data_rho_0.0.npz",
    test_fraction=0.2,
    seed=123,
):
    """
    Loads the linreg dataset and splits into train/test sets.
    Returns X_train, X_test, y_train, y_test, plus metadata dict.
    """
    data = np.load(path)
    X = data["X"]
    y = data["y"]

    N = X.shape[0]
    np.random.seed(seed)

    # --- Random permutation of indices ---
    idx = np.random.permutation(N)
    test_size = int(test_fraction * N)

    test_idx = idx[:test_size]
    train_idx = idx[test_size:]

    # --- Split ---
    X_train = X[train_idx]
    X_test = X[test_idx]
    y_train = y[train_idx]
    y_test = y[test_idx]

    meta = {key: data[key] for key in data.files if key not in ["X", "y"]}

    return X_train, X_test, y_train, y_test, meta, X, y

