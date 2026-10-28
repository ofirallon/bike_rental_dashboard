import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge, Lasso, PoissonRegressor

# 1. Seasonal Baseline Model
def fit_seasonal_baseline(df_train):
    """
    Learns the average rental count for each (hour, workingday) combination.
    """
    print("Fitting Seasonal Baseline...")
    df_train = df_train.copy()
    # Extract hour from timestamp
    df_train['hour'] = df_train['timestamp'].dt.hour
    
    # Calculate means grouped by hour and workingday
    lookup = df_train.groupby(['hour', 'workingday'])['cnt'].mean().to_dict()
    global_mean = df_train['cnt'].mean()
    
    return {"lookup": lookup, "global_mean": global_mean}

def predict_seasonal_baseline(baseline_model, timestamp, workingday):
    """ Returns the baseline prediction for a given timestamp and workingday. """
    hour = timestamp.hour
    key = (hour, workingday)
    # Return specific mean, or global mean if combination wasn't in training
    return baseline_model["lookup"].get(key, baseline_model["global_mean"])


# Helper function to separate Features (X) and Target (y)
def get_X_y(df):
    drop_cols = ['timestamp', 'period', 'cnt']
    X = df.drop(columns=drop_cols).astype(float)
    y = df['cnt'].astype(float)
    return X, y


# 2. Ridge & Lasso Regression
def fit_ridge(X_train, y_train, alpha=1.0):
    model = Ridge(alpha=alpha)
    model.fit(X_train, y_train)
    return model

def fit_lasso(X_train, y_train, alpha=1.0):
    model = Lasso(alpha=alpha)
    model.fit(X_train, y_train)
    return model


# 3. Poisson Count Forecast
def fit_poisson(X_train, y_train, alpha=1.0):
    model = PoissonRegressor(alpha=alpha)
    model.fit(X_train, y_train)
    return model


# 4. Bayesian Linear Regression
def fit_bayesian(X_train, y_train, alpha_prior=1.0, sigma2_noise=1.0):
    """
    Fits Bayesian Linear Regression.
    Calculates posterior mean (m) and covariance (S) matrices exactly as defined.
    """
    # Add an intercept column of ones at the beginning
    Phi = np.c_[np.ones(X_train.shape[0]), X_train.to_numpy()]
    y = y_train.to_numpy()
    d = Phi.shape[1]
    
    # I is the identity matrix
    I = np.eye(d)
    
    # Calculate matrix A: A = alpha * I + (Phi^T * Phi) / sigma^2
    A = (alpha_prior * I) + (Phi.T @ Phi) / sigma2_noise
    
    # Solve A * m = (Phi^T * y) / sigma^2
    right_side = (Phi.T @ y) / sigma2_noise
    m = np.linalg.solve(A, right_side)
    
    # Solve A * S = I (This avoids computing the inverse directly as requested in the project)
    S = np.linalg.solve(A, I)
    
    return {"m": m, "S": S, "alpha": alpha_prior, "sigma2": sigma2_noise}


if __name__ == "__main__":
    print("models.py is set up and ready to be imported!")