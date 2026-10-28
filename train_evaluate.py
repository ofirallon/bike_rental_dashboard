import pandas as pd
import numpy as np
import json
import warnings
import time
from scipy.stats import poisson
warnings.filterwarnings('ignore') # Ignore non-critical warnings

from data import audit_and_prepare_data
from features import build_features, fit_preprocessing, transform_features
from models import (fit_seasonal_baseline, predict_seasonal_baseline, get_X_y,
                    fit_ridge, fit_lasso, fit_poisson, fit_bayesian)

def calculate_mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def calculate_rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred)**2))

def run_pipeline():
    print("Starting training and evaluation pipeline...")
    
    # 1. Data Prep
    df = audit_and_prepare_data()
    df_featured, hist_cols = build_features(df)
    fit_preprocessing(df_featured, hist_cols)
    df_final = transform_features(df_featured, hist_cols)
    
    train_df = df_final[df_final['period'] == 'Training']
    val_df = df_final[df_final['period'] == 'Validation']
    test_df = df_final[df_final['period'] == 'Test']
    
    X_train, y_train = get_X_y(train_df)
    X_val, y_val = get_X_y(val_df)
    X_test, y_test = get_X_y(test_df)
    
    validation_results = []
    models_dict = {}
    
    # --- 1. Seasonal Baseline ---
    print("Evaluating Seasonal Baseline...")
    start_t = time.time()
    baseline_model = fit_seasonal_baseline(df[df['period'] == 'Training'])
    baseline_time = time.time() - start_t
    val_preds = [predict_seasonal_baseline(baseline_model, row['timestamp'], row['workingday']) for _, row in val_df.iterrows()]
    baseline_mae = calculate_mae(y_val.values, np.array(val_preds))
    validation_results.append({"Model": "Seasonal Baseline", "Parameters": "None", "MAE": baseline_mae})
    models_dict['Seasonal Baseline'] = {"model": baseline_model, "time": baseline_time}

    # --- 2. Ridge Regression ---
    print("Evaluating Ridge Regression...")
    start_t = time.time()
    best_ridge_mae = float('inf'); best_ridge_model = None; best_ridge_alpha = None
    for alpha in [0.1, 1.0, 10.0]:
        m = fit_ridge(X_train, y_train, alpha)
        p = np.maximum(m.predict(X_val), 0)
        cur_mae = calculate_mae(y_val, p)
        validation_results.append({"Model": "Ridge", "Parameters": f"alpha={alpha}", "MAE": cur_mae})
        if cur_mae < best_ridge_mae:
            best_ridge_mae = cur_mae; best_ridge_model = m; best_ridge_alpha = alpha
    models_dict['Ridge'] = {"model": best_ridge_model, "time": time.time() - start_t, "alpha": best_ridge_alpha}

    # --- 3. Lasso Regression ---
    print("Evaluating Lasso Regression...")
    start_t = time.time()
    best_lasso_mae = float('inf'); best_lasso_model = None; best_lasso_alpha = None
    for alpha in [0.01, 0.1, 0.5, 1.0, 5.0]:
        m = fit_lasso(X_train, y_train, alpha)
        p = np.maximum(m.predict(X_val), 0)
        cur_mae = calculate_mae(y_val, p)
        validation_results.append({"Model": "Lasso", "Parameters": f"alpha={alpha}", "MAE": cur_mae})
        if cur_mae < best_lasso_mae:
            best_lasso_mae = cur_mae; best_lasso_model = m; best_lasso_alpha = alpha
    models_dict['Lasso'] = {"model": best_lasso_model, "time": time.time() - start_t, "alpha": best_lasso_alpha}

    # --- 4. Poisson Regression ---
    print("Evaluating Poisson Regression...")
    start_t = time.time()
    best_poisson_mae = float('inf'); best_poisson_model = None; best_poisson_alpha = None
    for alpha in [0.01, 0.1, 1.0]:
        m = fit_poisson(X_train, y_train, alpha)
        p = m.predict(X_val)
        cur_mae = calculate_mae(y_val, p)
        validation_results.append({"Model": "Poisson", "Parameters": f"alpha={alpha}", "MAE": cur_mae})
        if cur_mae < best_poisson_mae:
            best_poisson_mae = cur_mae; best_poisson_model = m; best_poisson_alpha = alpha
    models_dict['Poisson'] = {"model": best_poisson_model, "time": time.time() - start_t, "alpha": best_poisson_alpha}

    # --- 5. Bayesian Linear Regression ---
    print("Evaluating Bayesian Regression...")
    start_t = time.time()
    best_bayesian_mae = float('inf'); best_bayesian_model = None; best_bayesian_config = None
    for alpha_prior, sigma2 in [(1.0, 1.0), (0.1, 1.0)]:
        m = fit_bayesian(X_train, y_train, alpha_prior, sigma2)
        Phi_val = np.c_[np.ones(X_val.shape[0]), X_val.to_numpy()]
        p = np.maximum(Phi_val @ m['m'], 0)
        cur_mae = calculate_mae(y_val, p)
        validation_results.append({"Model": "Bayesian", "Parameters": f"prior={alpha_prior}, noise={sigma2}", "MAE": cur_mae})
        if cur_mae < best_bayesian_mae:
            best_bayesian_mae = cur_mae; best_bayesian_model = m; best_bayesian_config = {"alpha": alpha_prior, "sigma2": sigma2}
    models_dict['Bayesian'] = {"model": best_bayesian_model, "time": time.time() - start_t, "config": best_bayesian_config}

    # --- Save validation_results.csv ---
    pd.DataFrame(validation_results).to_csv("validation_results.csv", index=False)
    print("-> Saved validation_results.csv")

    # --- Determine WINNER (Default Model) ---
    best_val_maes = {
        'Seasonal Baseline': baseline_mae, 'Ridge': best_ridge_mae, 
        'Lasso': best_lasso_mae, 'Poisson': best_poisson_mae, 'Bayesian': best_bayesian_mae
    }
    best_model_name = min(best_val_maes, key=best_val_maes.get)
    
    print("\n" + "="*50)
    print(f" WINNER (Default Model): {best_model_name} ")
    print("="*50 + "\n")

    # Update config.json
    with open("config.json", "r") as f:
        config = json.load(f)
    config["default_model"] = best_model_name
    with open("config.json", "w") as f:
        json.dump(config, f, indent=4)

    # --- TEST SET EVALUATION & PREDICTIONS CSV ---
    print("Evaluating models on Final Test Set and saving predictions...")
    predictions_list = []
    metrics_list = []

    for model_name in ['Seasonal Baseline', 'Ridge', 'Lasso', 'Poisson', 'Bayesian']:
        m = models_dict[model_name]['model']
        preds_raw = []; preds_scored = []
        intervals = {'lower_80': [], 'upper_80': [], 'lower_95': [], 'upper_95': [],
                     'obs_var': [], 'coef_var': [], 'tot_var': []}

        if model_name == 'Seasonal Baseline':
            for _, row in test_df.iterrows():
                p = predict_seasonal_baseline(m, row['timestamp'], row['workingday'])
                preds_raw.append(p); preds_scored.append(p)
                for k in intervals: intervals[k].append(np.nan)

        elif model_name in ['Ridge', 'Lasso']:
            p_raw = m.predict(X_test)
            preds_raw = p_raw.tolist()
            preds_scored = np.maximum(p_raw, 0).tolist()
            for _ in range(len(test_df)):
                for k in intervals: intervals[k].append(np.nan)

        elif model_name == 'Poisson':
            p_raw = m.predict(X_test)
            preds_raw = p_raw.tolist(); preds_scored = p_raw.tolist()
            for mu in p_raw:
                intervals['lower_80'].append(poisson.ppf(0.10, mu))
                intervals['upper_80'].append(poisson.ppf(0.90, mu))
                intervals['lower_95'].append(poisson.ppf(0.025, mu))
                intervals['upper_95'].append(poisson.ppf(0.975, mu))
                intervals['obs_var'].append(np.nan); intervals['coef_var'].append(np.nan); intervals['tot_var'].append(np.nan)

        elif model_name == 'Bayesian':
            Phi_test = np.c_[np.ones(X_test.shape[0]), X_test.to_numpy()]
            p_raw = Phi_test @ m['m']
            preds_raw = p_raw.tolist()
            preds_scored = np.maximum(p_raw, 0).tolist()
            sigma2 = models_dict['Bayesian']['config']['sigma2']
            S = m['S']
            for i in range(len(test_df)):
                phi = Phi_test[i]
                coef_var = phi.T @ S @ phi
                tot_var = sigma2 + coef_var
                std_dev = np.sqrt(tot_var)
                mu = p_raw[i]
                intervals['obs_var'].append(sigma2); intervals['coef_var'].append(coef_var); intervals['tot_var'].append(tot_var)
                intervals['lower_80'].append(mu - 1.28155 * std_dev); intervals['upper_80'].append(mu + 1.28155 * std_dev)
                intervals['lower_95'].append(mu - 1.95996 * std_dev); intervals['upper_95'].append(mu + 1.95996 * std_dev)

        # Build prediction DataFrame for this model
        model_df = test_df[['timestamp', 'period', 'cnt']].copy()
        model_df['model'] = model_name
        model_df['raw_mean'] = preds_raw
        model_df['scored_forecast'] = preds_scored
        model_df['error'] = model_df['cnt'] - model_df['scored_forecast']
        model_df['abs_error'] = np.abs(model_df['error'])
        for k in intervals: model_df[k] = intervals[k]
        
        predictions_list.append(model_df)

        # Calculate final metrics
        mae = calculate_mae(y_test, np.array(preds_scored))
        rmse = calculate_rmse(y_test, np.array(preds_scored))
        metrics_list.append({
            "Model": model_name,
            "MAE": mae,
            "RMSE": rmse,
            "Evaluated_Hours": len(test_df),
            "Training_Time_sec": models_dict[model_name]['time']
        })

    # Save to CSV
    final_preds_df = pd.concat(predictions_list, ignore_index=True)
    final_preds_df.to_csv("predictions.csv", index=False)
    print("-> Saved predictions.csv")
    
    pd.DataFrame(metrics_list).to_csv("metrics.csv", index=False)
    print("-> Saved metrics.csv")
    
    print("\nPipeline finished completely! All files are ready.")

if __name__ == "__main__":
    run_pipeline()