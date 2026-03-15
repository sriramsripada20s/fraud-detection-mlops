"""
train.py — XGBoost training pipeline with MLflow experiment tracking

What this does:
  1. Loads IEEE-CIS data with memory optimisation
  2. Splits raw data FIRST (prevents leakage)
  3. Engineers features on train only, applies to val
  4. Trains XGBoost with class imbalance handling
  5. Tunes threshold using F2-score
  6. Compares results against baseline notebook numbers
  7. Saves model + preprocessing artifacts
  8. Logs everything to MLflow

Targets to beat (from notebooks/04_baseline_model.ipynb):
  avg_precision : 0.6257
  recall_fraud  : 0.6747
  fraud_caught  : 67.47%

Run:
  cd src
  python train.py
  python train.py ../data/raw    # custom data dir

MLflow UI:
  mlflow ui --port 5000
  open http://localhost:5000
"""

import os
import sys
import gc
import json
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
import mlflow
import mlflow.xgboost
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv

# Add src/ to path so we can import sibling modules
sys.path.insert(0, os.path.dirname(__file__))
from preprocess import (
    load_data,
    engineer_features,
    get_feature_cols,
    save_preprocessing_artifacts,
)
from evaluate import find_best_threshold, full_evaluation

load_dotenv()

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------
MLFLOW_EXPERIMENT = 'fraud-detection'
MODEL_DIR         = os.getenv('MODEL_DIR',          '../model')
DATA_DIR          = os.getenv('DATA_DIR',            '../data/raw')
MLFLOW_URI        = os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000')
PROCESSED_DIR     = '../data/processed'

# XGBoost params
# Key differences from baseline notebook:
#   - More estimators (600 vs 300)
#   - Lower learning rate (0.05 vs 0.1) — slower but better generalisation
#   - Added regularisation: min_child_weight, gamma, reg_alpha
#   - eval_metric = aucpr (area under PR curve — better for imbalance)
XGBOOST_PARAMS = {
    'n_estimators':          600,
    'max_depth':             6,
    'learning_rate':         0.05,
    'subsample':             0.8,
    'colsample_bytree':      0.8,
    'min_child_weight':      10,    # prevents overfitting on rare fraud patterns
    'gamma':                 1,     # minimum loss reduction to split
    'reg_alpha':             0.1,   # L1 regularisation
    'reg_lambda':            1.0,   # L2 regularisation
    'eval_metric':           'aucpr',
    'early_stopping_rounds': 50,
    'random_state':          42,
    'n_jobs':                -1,
    'verbosity':             1,
}


# ------------------------------------------------------------------
# Main training function
# ------------------------------------------------------------------
def train(data_dir: str = DATA_DIR) -> str:
    """
    Run full training pipeline.
    Returns MLflow run_id.
    """
    # Try to connect to MLflow — fall back to local if not running
    try:
        mlflow.set_tracking_uri(MLFLOW_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT)
        use_mlflow = True
        print(f"MLflow tracking : {MLFLOW_URI}")
    except Exception:
        use_mlflow = False
        print("MLflow not running — saving artifacts locally only")
        print("Start MLflow with: mlflow ui --port 5000")

    run_id = None

    with mlflow.start_run() if use_mlflow else _dummy_context() as run:

        if use_mlflow:
            run_id = run.info.run_id
            print(f"MLflow run ID   : {run_id}\n")

        # ----------------------------------------------------------
        # 1. Load data
        # ----------------------------------------------------------
        print("="*55)
        print("  STEP 1 — Loading data")
        print("="*55)
        df = load_data(data_dir)

        # ----------------------------------------------------------
        # 2. Split BEFORE feature engineering — prevents leakage
        #    If we engineer first then split, val data leaks into
        #    freq_maps and label encoders
        # ----------------------------------------------------------
        print("\n" + "="*55)
        print("  STEP 2 — Stratified train/val split")
        print("="*55)

        X_raw = df.drop(columns=['isFraud'])
        y     = df['isFraud']
        del df; gc.collect()

        # First split — carve out 15% test set (never seen during training)
        X_trainval, X_test_raw, y_trainval, y_test = train_test_split(
            X_raw, y,
            test_size=0.15,
            stratify=y,
            random_state=42,
        )

        # Second split — remaining 85% split into 70% train / 15% val
        X_tr_raw, X_val_raw, y_train, y_val = train_test_split(
            X_trainval, y_trainval,
            test_size=0.176,   # 0.176 x 0.85 ≈ 0.15 of total
            stratify=y_trainval,
            random_state=42,
        )
        del X_raw, X_trainval, y_trainval; gc.collect()

        print(f"Train : {X_tr_raw.shape[0]:,} rows "
              f"| fraud: {y_train.mean()*100:.2f}%")
        print(f"Val   : {X_val_raw.shape[0]:,} rows "
              f"| fraud: {y_val.mean()*100:.2f}%")
        print(f"Test  : {X_test_raw.shape[0]:,} rows "
              f"| fraud: {y_test.mean()*100:.2f}%")

        # ----------------------------------------------------------
        # 3. Feature engineering
        #    fit=True on train — computes freq_maps, label encoders
        #    fit=False on val  — applies pre-computed maps (no leakage)
        # ----------------------------------------------------------
        print("\n" + "="*55)
        print("  STEP 3 — Feature engineering")
        print("="*55)

        tr_df        = X_tr_raw.copy()
        tr_df['isFraud'] = y_train.values
        val_df       = X_val_raw.copy()
        val_df['isFraud'] = y_val.values
        del X_tr_raw, X_val_raw; gc.collect()

        print("Engineering train features (fit=True)...")
        train_eng, freq_maps, label_encoders = engineer_features(
            tr_df, fit=True
        )
        del tr_df; gc.collect()

        print("Engineering val features (fit=False)...")
        val_eng, _, _ = engineer_features(
            val_df, freq_maps, label_encoders, fit=False
        )
        del val_df; gc.collect()

        feature_cols = get_feature_cols(train_eng)
        X_train      = train_eng[feature_cols]
        X_val        = val_eng[feature_cols]
        del train_eng, val_eng; gc.collect()

        print(f"Feature count : {len(feature_cols)}")

        # Validate no nulls
        assert X_train.isnull().sum().sum() == 0, \
            "Nulls in training data — check preprocess.py"
        assert X_val.isnull().sum().sum() == 0, \
            "Nulls in val data — check preprocess.py"
        print("Null check    : PASSED")

        # ----------------------------------------------------------
        # 4. Class imbalance — auto compute scale_pos_weight
        #    EDA: 3.5% fraud = 28:1 ratio
        #    scale_pos_weight tells XGBoost to penalise missing
        #    a fraud transaction 28x more than missing a legit one
        # ----------------------------------------------------------
        print("\n" + "="*55)
        print("  STEP 4 — Class imbalance handling")
        print("="*55)

        neg = int((y_train == 0).sum())
        pos = int((y_train == 1).sum())
        spw = round(neg / pos, 2)
        print(f"Legit : {neg:,}")
        print(f"Fraud : {pos:,}")
        print(f"scale_pos_weight = {spw}")

        params = {**XGBOOST_PARAMS, 'scale_pos_weight': spw}

        # Log params to MLflow
        if use_mlflow:
            mlflow.log_params({
                **{k: v for k, v in params.items()
                   if k not in ('early_stopping_rounds',)},
                'early_stopping_rounds': params['early_stopping_rounds'],
                'n_train':               X_train.shape[0],
                'n_val':                 X_val.shape[0],
                'n_features':            len(feature_cols),
                'fraud_rate_train':      round(float(y_train.mean()), 4),
            })

        # ----------------------------------------------------------
        # 5. Train
        # ----------------------------------------------------------
        print("\n" + "="*55)
        print("  STEP 5 — Training XGBoost")
        print("="*55)
        print("(early stopping on val aucpr — patience=50)\n")

        model = xgb.XGBClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=100,
        )

        print(f"\nBest iteration : {model.best_iteration}")

        # ----------------------------------------------------------
        # 6. Threshold tuning + evaluation
        # ----------------------------------------------------------
        print("\n" + "="*55)
        print("  STEP 6 — Threshold tuning + evaluation")
        print("="*55)

        val_proba   = model.predict_proba(X_val)[:, 1]
        train_proba = model.predict_proba(X_train)[:, 1]

        best_threshold, thr_metrics = find_best_threshold(
            y_val.values, val_proba, beta=2.0
        )
        print(f"Optimal threshold (F2): {best_threshold:.4f}")

        os.makedirs(PROCESSED_DIR, exist_ok=True)

        val_metrics = full_evaluation(
            y_val.values, val_proba,
            threshold=best_threshold,
            save_dir=PROCESSED_DIR,
            split_name='val',
        )

        train_metrics = full_evaluation(
            y_train.values, train_proba,
            threshold=best_threshold,
            save_dir=PROCESSED_DIR,
            split_name='train',
        )

        # Overfitting check — train after both are defined
        ap_gap = train_metrics['avg_precision'] - val_metrics['avg_precision']
        print(f"Overfit gap (train AP - val AP): {ap_gap:.4f}")
        if ap_gap > 0.15:
            print("WARNING: Possible overfitting — consider more regularisation")
        
        # Evaluate on held-out test set — truly unseen data
        print("\n" + "="*55)
        print("  STEP 6b — Test set evaluation (held-out)")
        print("="*55)

        # Engineer test features using train's freq_maps/label_encoders
        test_df           = X_test_raw.copy()
        test_df['isFraud']= y_test.values
        test_eng, _, _    = engineer_features(
            test_df, freq_maps, label_encoders, fit=False
        )
        del test_df, X_test_raw; gc.collect()

        X_test      = test_eng[feature_cols]
        test_proba  = model.predict_proba(X_test)[:, 1]

        test_metrics = full_evaluation(
            y_test.values, test_proba,
            threshold=best_threshold,
            save_dir=PROCESSED_DIR,
            split_name='test',
        )

        if use_mlflow:
            for k, v in test_metrics.items():
                if isinstance(v, (int, float)):
                    mlflow.log_metric(f'test_{k}', v)

        # Save test metrics
        with open(os.path.join(PROCESSED_DIR, 'test_results.json'), 'w') as f:
            json.dump(test_metrics, f, indent=2)

        # Overfitting check
        ap_gap = train_metrics['avg_precision'] - val_metrics['avg_precision']
        print(f"Overfit gap (train AP - val AP): {ap_gap:.4f}")
        if ap_gap > 0.15:
            print("WARNING: Possible overfitting — consider more regularisation")

        # Log metrics to MLflow
        if use_mlflow:
            for k, v in val_metrics.items():
                if isinstance(v, (int, float)):
                    mlflow.log_metric(f'val_{k}', v)
            for k, v in train_metrics.items():
                if isinstance(v, (int, float)):
                    mlflow.log_metric(f'train_{k}', v)

        # ----------------------------------------------------------
        # 7. Feature importance
        # ----------------------------------------------------------
        print("\n" + "="*55)
        print("  STEP 7 — Feature importance")
        print("="*55)

        fi      = pd.Series(model.feature_importances_, index=feature_cols)
        fi_top  = fi.sort_values(ascending=False).head(30)
        fi_dict = fi_top.to_dict()

        print("Top 10 features:")
        for feat, score in list(fi_dict.items())[:10]:
            print(f"  {feat:<35} {score:.4f}")

        if use_mlflow:
            mlflow.log_dict(fi_dict, 'feature_importance.json')

        # ----------------------------------------------------------
        # 8. Save artifacts
        # ----------------------------------------------------------
        print("\n" + "="*55)
        print("  STEP 8 — Saving artifacts")
        print("="*55)

        os.makedirs(MODEL_DIR, exist_ok=True)
        model_path = os.path.join(MODEL_DIR, 'model.joblib')

        artifact = {
            'model':           model,
            'feature_cols':    feature_cols,
            'threshold':       best_threshold,
            'freq_maps':       freq_maps,
            'label_encoders':  label_encoders,
            'val_metrics':     val_metrics,
            'xgboost_params':  {
                k: v for k, v in params.items() if not callable(v)
            },
        }
        joblib.dump(artifact, model_path)
        print(f"Model saved     → {model_path}")

        save_preprocessing_artifacts(
            freq_maps, label_encoders, feature_cols,
            path=os.path.join(MODEL_DIR, 'preprocessing.joblib')
        )

        # Save val metrics as JSON for reference
        metrics_path = os.path.join(PROCESSED_DIR, 'production_results.json')
        with open(metrics_path, 'w') as f:
            json.dump(val_metrics, f, indent=2)
        print(f"Metrics saved   → {metrics_path}")

        if use_mlflow:
            mlflow.xgboost.log_model(model, artifact_path='xgboost-model')
            mlflow.log_artifact(model_path)
            print(f"\nMLflow UI → {MLFLOW_URI}")
            print(f"Run ID    → {run_id}")

        print("\n" + "="*55)
        print("  TRAINING COMPLETE")
        print("="*55)

    return run_id


# ------------------------------------------------------------------
# Dummy context manager — used when MLflow is not running
# ------------------------------------------------------------------
class _dummy_context:
    def __enter__(self): return self
    def __exit__(self, *args): pass


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------
if __name__ == '__main__':
    data_dir = sys.argv[1] if len(sys.argv) > 1 else DATA_DIR
    train(data_dir)