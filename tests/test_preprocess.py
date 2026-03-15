"""
test_preprocess.py — Critical tests for feature engineering

Only testing what matters most:
  1. No nulls after engineering (model crashes if nulls exist)
  2. No leakage in freq maps (inflated metrics if leakage exists)
  3. Feature columns match train vs val (shape error at inference)
  4. Velocity features computed (card1_fraud_rate is top feature)
  5. Unseen categories don't crash (production safety)

Run:
  pytest tests/test_preprocess.py -v
"""

import sys
import os
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from preprocess import engineer_features, get_feature_cols


# ------------------------------------------------------------------
# Fixture — minimal synthetic data
# ------------------------------------------------------------------
def make_df(n=300, seed=42):
    rng     = np.random.default_rng(seed)
    n_fraud = max(1, int(n * 0.035))
    return pd.DataFrame({
        'TransactionID':  range(n),
        'TransactionDT':  rng.integers(86400, 86400*180, n),
        'TransactionAmt': rng.exponential(150, n).clip(1, 10000),
        'isFraud':        [1]*n_fraud + [0]*(n - n_fraud),
        'ProductCD':      rng.choice(['W', 'H', 'C'], n),
        'card1':          rng.integers(1000, 9999, n).astype(float),
        'card4':          rng.choice(['visa', 'mastercard', None], n),
        'card6':          rng.choice(['debit', 'credit', None], n),
        'addr1':          rng.integers(100, 500, n).astype(float),
        'addr2':          rng.integers(1, 100, n).astype(float),
        'P_emaildomain':  rng.choice(['gmail.com', 'yahoo.com', None], n),
        'DeviceType':     rng.choice(['desktop', 'mobile', None], n),
        'DeviceInfo':     rng.choice(['Windows', 'iOS', None], n),
        'id_01':          np.where(rng.random(n) > 0.7, rng.random(n), np.nan),
        'M1':             rng.choice(['T', 'F', None], n),
        'V1':             np.where(rng.random(n) > 0.3, rng.random(n), np.nan),
        'C1':             rng.integers(0, 10, n).astype(float),
        'D1':             rng.integers(0, 100, n).astype(float),
    })


@pytest.fixture
def split_dfs():
    df    = make_df(n=300)
    train = df.iloc[:240].copy()
    val   = df.iloc[240:].copy()
    return train, val


# ------------------------------------------------------------------
# Test 1 — No nulls (model crashes if nulls exist)
# ------------------------------------------------------------------
def test_no_nulls_after_engineering(split_dfs):
    train, val = split_dfs
    train_eng, freq_maps, les = engineer_features(train, fit=True)
    val_eng,   _,          _  = engineer_features(val, freq_maps, les, fit=False)

    fcols = get_feature_cols(train_eng)
    assert train_eng[fcols].isnull().sum().sum() == 0, "Nulls in train"
    assert val_eng[fcols].isnull().sum().sum()   == 0, "Nulls in val"


# ------------------------------------------------------------------
# Test 2 — No leakage (inflated metrics if freq maps leak)
# ------------------------------------------------------------------
def test_no_leakage_freq_maps(split_dfs):
    train, val = split_dfs
    train_eng, freq_maps, les = engineer_features(train, fit=True)

    # Inject unseen card1 into val
    val_modified = val.copy()
    val_modified.iloc[0, val_modified.columns.get_loc('card1')] = 999999.0

    val_eng, _, _ = engineer_features(val_modified, freq_maps, les, fit=False)

    # Unseen card1 must get freq=0 not a real count
    assert val_eng.iloc[0]['card1_freq'] == 0, "Leakage detected in freq maps"


# ------------------------------------------------------------------
# Test 3 — Feature columns match train vs val (inference shape error)
# ------------------------------------------------------------------
def test_feature_cols_match_train_val(split_dfs):
    train, val = split_dfs
    train_eng, freq_maps, les = engineer_features(train, fit=True)
    val_eng,   _,          _  = engineer_features(val, freq_maps, les, fit=False)

    train_fcols = get_feature_cols(train_eng)
    val_fcols   = get_feature_cols(val_eng)
    assert train_fcols == val_fcols, \
        f"Column mismatch: {set(train_fcols) ^ set(val_fcols)}"


# ------------------------------------------------------------------
# Test 4 — Velocity features exist (card1_fraud_rate is top feature)
# ------------------------------------------------------------------
def test_velocity_features_created(split_dfs):
    train, _ = split_dfs
    train_eng, _, _ = engineer_features(train, fit=True)

    for col in ['card1_fraud_rate', 'card1_txn_count',
                'amt_vs_card_avg', 'hour_fraud_rate']:
        assert col in train_eng.columns, f"Missing velocity feature: {col}"

    assert (train_eng['card1_fraud_rate'] >= 0).all()
    assert (train_eng['card1_fraud_rate'] <= 1).all()


# ------------------------------------------------------------------
# Test 5 — Unseen categories don't crash (production safety)
# ------------------------------------------------------------------
def test_unseen_category_does_not_crash(split_dfs):
    train, val = split_dfs
    train_eng, freq_maps, les = engineer_features(train, fit=True)

    val_new_cat = val.copy()
    val_new_cat.iloc[0, val_new_cat.columns.get_loc('ProductCD')] = 'XXXX'

    val_eng, _, _ = engineer_features(val_new_cat, freq_maps, les, fit=False)
    assert val_eng['ProductCD'].isnull().sum() == 0