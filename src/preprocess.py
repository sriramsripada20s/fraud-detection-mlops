"""
preprocess.py — IEEE-CIS Fraud Detection feature engineering

Design principles:
  - Training uses ALL features including vendor-provided V/C/D columns
  - API schema exposes only understandable real-time fields
  - V/C/D columns default to 0 at inference if vendor not integrated
  - Leak-free: freq_maps and label_encoders computed on train only
  - Memory optimised for 8GB RAM machines

Performance:
  With vendor features (V/C/D provided) : AP = 0.7162
  Without vendor features (V/C/D = 0)   : AP = 0.5484
  Gap documented honestly in README

Feature groups:
  - Transaction : amount (log1p, decimal, round), time (hour, is_night)
  - Card        : card1-card6, frequency encoding, velocity aggregations
  - Address     : addr1, addr2, frequency encoding
  - Email       : P_emaildomain, R_emaildomain
  - Device      : DeviceType, DeviceInfo, id_30, id_31, id_33
  - Identity    : id_01-id_11 with missingness flags
  - Match flags : M1-M9
  - Vendor      : V1-V339, C1-C14, D1-D15 (optional, default 0)
  - Engineered  : card1_fraud_rate, amt_vs_card_avg, hour_fraud_rate
"""

import os
import sys
import gc
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import joblib
from typing import Optional

# ------------------------------------------------------------------
# Column group definitions
# ------------------------------------------------------------------
CAT_COLS = [
    'ProductCD', 'card4', 'card6',
    'P_emaildomain', 'R_emaildomain',
    'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9',
    'DeviceType', 'DeviceInfo',
]

ID_CAT_COLS = [
    'id_12', 'id_15', 'id_16', 'id_23', 'id_27', 'id_28',
    'id_29', 'id_30', 'id_31', 'id_32', 'id_33', 'id_34',
    'id_35', 'id_36', 'id_37', 'id_38',
]

# High cardinality → frequency encode
FREQ_COLS = ['card1', 'card2', 'addr1', 'addr2']

# Vendor provided columns — included in training, default 0 at inference
V_COLS = [f'V{i}' for i in range(1, 340)]
C_COLS = [f'C{i}' for i in range(1, 15)]
D_COLS = [f'D{i}' for i in range(1, 16)]

# Numeric identity block
ID_NUM_COLS = [f'id_{str(i).zfill(2)}' for i in range(1, 12)]

# Drop after feature extraction
DROP_COLS = ['TransactionID', 'TransactionDT']


# ------------------------------------------------------------------
# I/O helpers
# ------------------------------------------------------------------
def load_data(data_dir: str) -> pd.DataFrame:
    """
    Load and merge transaction + identity tables.
    Reads V-columns as float32 to manage memory on 8GB machines.
    """
    v_dtypes = {f'V{i}': 'float32' for i in range(1, 340)}
    num_dtypes = {
        'TransactionAmt': 'float32',
        'card1': 'float32', 'card2': 'float32',
        'card3': 'float32', 'card5': 'float32',
        'addr1': 'float32', 'addr2': 'float32',
        'dist1': 'float32', 'dist2': 'float32',
    }

    txn = pd.read_csv(
        os.path.join(data_dir, 'train_transaction.csv'),
        dtype={**v_dtypes, **num_dtypes}
    )
    idn = pd.read_csv(os.path.join(data_dir, 'train_identity.csv'))
    df  = txn.merge(idn, on='TransactionID', how='left')

    del txn, idn
    gc.collect()

    print(f"Loaded   : {df.shape[0]:,} rows x {df.shape[1]} cols")
    print(f"Fraud    : {df['isFraud'].mean()*100:.2f}%")
    print(f"Memory   : {df.memory_usage(deep=True).sum()/1e6:.0f} MB")
    return df


def load_test_data(data_dir: str) -> pd.DataFrame:
    """Load and merge test tables (no isFraud column)."""
    v_dtypes = {f'V{i}': 'float32' for i in range(1, 340)}
    txn = pd.read_csv(
        os.path.join(data_dir, 'test_transaction.csv'),
        dtype=v_dtypes
    )
    idn = pd.read_csv(os.path.join(data_dir, 'test_identity.csv'))
    df  = txn.merge(idn, on='TransactionID', how='left')
    del txn, idn
    gc.collect()
    return df


# ------------------------------------------------------------------
# Core feature engineering
# ------------------------------------------------------------------
def engineer_features(
    df: pd.DataFrame,
    freq_maps: Optional[dict]      = None,
    label_encoders: Optional[dict] = None,
    fit: bool = True,
) -> tuple:
    """
    Engineer features from raw merged dataframe.

    Parameters
    ----------
    df             : merged transaction + identity dataframe
    freq_maps      : pre-computed frequency maps — pass None when fit=True
    label_encoders : pre-fitted LabelEncoders — pass None when fit=True
    fit            : True  = compute maps from df (training data only)
                     False = apply pre-computed maps (val / test / inference)

    Returns
    -------
    (df_engineered, freq_maps, label_encoders)
    """
    df = df.copy()

    if freq_maps      is None: freq_maps      = {}
    if label_encoders is None: label_encoders = {}

    # ----------------------------------------------------------
    # 1. Time features
    #    EDA finding: night hours show elevated fraud rate
    # ----------------------------------------------------------
    if 'TransactionDT' in df.columns:
        df['hour']        = (df['TransactionDT'] // 3600) % 24
        df['day_of_week'] = (df['TransactionDT'] // 86400) % 7
        df['is_night']    = ((df['hour'] >= 22) | (df['hour'] <= 5)).astype(int)
        df['is_weekend']  = (df['day_of_week'] >= 5).astype(int)

    # ----------------------------------------------------------
    # 2. Amount features
    #    EDA finding: right-skewed, decimal part is informative
    # ----------------------------------------------------------
    if 'TransactionAmt' in df.columns:
        df['amt_log']      = np.log1p(df['TransactionAmt'])
        df['amt_decimal']  = df['TransactionAmt'] % 1
        df['amt_cents']    = (df['TransactionAmt'] * 100).astype('int64') % 100
        df['amt_is_round'] = (df['amt_decimal'] == 0).astype(int)

    # ----------------------------------------------------------
    # 3. Frequency encoding — LEAKAGE FREE
    #    fit=True  → compute counts from training data only
    #    fit=False → map using pre-computed freq_maps
    # ----------------------------------------------------------
    for col in FREQ_COLS:
        if col not in df.columns:
            continue
        if fit:
            freq_maps[col] = df[col].value_counts().to_dict()
        df[f'{col}_freq'] = (
            df[col].map(freq_maps.get(col, {}))
                   .fillna(0)
                   .astype(int)
        )

    # ----------------------------------------------------------
    # 3b. Velocity features — card behaviour aggregations
    #     Replaces V-columns with interpretable business features
    #     Computed on training data only (fit=True) — leak-free
    #     EDA motivation: card frequency showed velocity signal
    # ----------------------------------------------------------
    if fit:
        agg_dict = {
            'card1_txn_count': ('TransactionAmt', 'count'),
            'card1_avg_amt':   ('TransactionAmt', 'mean'),
            'card1_max_amt':   ('TransactionAmt', 'max'),
        }
        if 'isFraud' in df.columns:
            agg_dict['card1_fraud_rate'] = ('isFraud', 'mean')

        card_stats = df.groupby('card1').agg(**agg_dict).reset_index()
        freq_maps['card1_stats'] = (
            card_stats.set_index('card1').to_dict('index')
        )

        if 'TransactionDT' in df.columns and 'isFraud' in df.columns:
            _hour      = (df['TransactionDT'] // 3600) % 24
            hour_fraud = df.groupby(_hour)['isFraud'].mean().to_dict()
            freq_maps['hour_fraud_rate'] = hour_fraud

    # Apply card stats
    card_stats_map = freq_maps.get('card1_stats', {})

    df['card1_txn_count'] = df['card1'].map(
        lambda x: card_stats_map.get(x, {}).get('card1_txn_count', 0)
    )
    df['card1_avg_amt'] = df['card1'].map(
        lambda x: card_stats_map.get(x, {}).get('card1_avg_amt', 0)
    )
    df['card1_max_amt'] = df['card1'].map(
        lambda x: card_stats_map.get(x, {}).get('card1_max_amt', 0)
    )
    df['card1_fraud_rate'] = df['card1'].map(
        lambda x: card_stats_map.get(x, {}).get('card1_fraud_rate', 0)
    )

    # Vectorized ratio features
    df['amt_vs_card_avg'] = (
        df['TransactionAmt'] / df['card1_avg_amt'].replace(0, 1)
    )
    df['amt_vs_card_max'] = (
        df['TransactionAmt'] / df['card1_max_amt'].replace(0, 1)
    )

    # Hour fraud rate
    if 'TransactionDT' in df.columns:
        _hour    = (df['TransactionDT'] // 3600) % 24
        hour_map = freq_maps.get('hour_fraud_rate', {})
        df['hour_fraud_rate'] = _hour.map(hour_map).fillna(0)

    # ----------------------------------------------------------
    # 4. Identity presence flag
    #    EDA finding: missing identity = 2x fraud rate
    # ----------------------------------------------------------
    df['has_identity'] = (
        df['id_01'].notnull().astype(int)
        if 'id_01' in df.columns else 0
    )

    # ----------------------------------------------------------
    # 5. Missingness flags for numeric identity block
    #    EDA finding: null is NOT random — fraudsters skip identity
    # ----------------------------------------------------------
    for col in ID_NUM_COLS:
        if col in df.columns:
            df[f'{col}_missing'] = df[col].isnull().astype(int)

    # ----------------------------------------------------------
    # 6. Null filling by column type
    # ----------------------------------------------------------

    # V-columns → fill 0
    # EDA: null = feature not triggered, 0 is correct
    # At inference: defaults to 0 if vendor not integrated
    v_present = [c for c in V_COLS if c in df.columns]
    df[v_present] = df[v_present].fillna(0)

    # C/D columns → fill 0
    c_present = [c for c in C_COLS if c in df.columns]
    d_present = [c for c in D_COLS if c in df.columns]
    df[c_present] = df[c_present].fillna(0)
    df[d_present] = df[d_present].fillna(0)

    # id numeric → fill -1
    id_num_present = [c for c in ID_NUM_COLS if c in df.columns]
    df[id_num_present] = df[id_num_present].fillna(-1)

    # Card / addr numerics → fill median
    for col in ['card1', 'card2', 'card3', 'card5']:
        if col in df.columns:
            if fit:
                freq_maps[f'{col}_median'] = float(df[col].median())
            df[col] = df[col].fillna(freq_maps.get(f'{col}_median', 0))

    # ----------------------------------------------------------
    # 7. Categorical encoding
    # ----------------------------------------------------------
    all_cat = CAT_COLS + ID_CAT_COLS
    for col in all_cat:
        if col not in df.columns:
            continue
        df[col] = df[col].fillna('unknown').astype(str)
        if fit:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            label_encoders[col] = le
        else:
            le = label_encoders.get(col)
            if le is not None:
                known = set(le.classes_)
                # Fall back to first known class if 'unknown' not in encoder
                fallback = 'unknown' if 'unknown' in known else le.classes_[0]
                df[col] = df[col].apply(
                    lambda x: x if x in known else fallback
                )
                df[col] = le.transform(df[col])
            else:
                df[col] = 0

    # ----------------------------------------------------------
    # 8. Fill remaining nulls with 0
    # ----------------------------------------------------------
    df = df.fillna(0)

    # ----------------------------------------------------------
    # 9. Safety net — encode any remaining object columns
    # ----------------------------------------------------------
    remaining_obj = [
        c for c in df.select_dtypes(include='object').columns
        if c != 'isFraud'
    ]
    if remaining_obj:
        print(f"Safety encoding {len(remaining_obj)} remaining object cols: {remaining_obj}")
        for col in remaining_obj:
            if fit:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                label_encoders[col] = le
            else:
                le = label_encoders.get(col)
                if le is not None:
                    known   = set(le.classes_)
                    fallback = 'unknown' if 'unknown' in known else le.classes_[0]
                    df[col] = df[col].astype(str).apply(
                        lambda x: x if x in known else fallback
                    )
                    df[col] = le.transform(df[col])
                else:
                    df[col] = 0

    # ----------------------------------------------------------
    # 10. Drop columns not needed for training
    # ----------------------------------------------------------
    df = df.drop(
        columns=[c for c in DROP_COLS if c in df.columns],
        errors='ignore'
    )

    return df, freq_maps, label_encoders


def get_feature_cols(df: pd.DataFrame) -> list:
    """Return model input columns (everything except target)."""
    return [c for c in df.columns if c not in ('isFraud', 'TransactionID')]


# ------------------------------------------------------------------
# Artifact persistence
# ------------------------------------------------------------------
def save_preprocessing_artifacts(
    freq_maps: dict,
    label_encoders: dict,
    feature_cols: list,
    path: str = 'model/preprocessing.joblib',
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump({
        'freq_maps':      freq_maps,
        'label_encoders': label_encoders,
        'feature_cols':   feature_cols,
    }, path)
    print(f"Preprocessing artifacts saved → {path}")


def load_preprocessing_artifacts(
    path: str = 'model/preprocessing.joblib'
) -> tuple:
    artifacts = joblib.load(path)
    return (
        artifacts['freq_maps'],
        artifacts['label_encoders'],
        artifacts['feature_cols'],
    )


# ------------------------------------------------------------------
# CLI entry point
# ------------------------------------------------------------------
if __name__ == '__main__':
    data_dir = sys.argv[1] if len(sys.argv) > 1 else '../data/raw'

    df = load_data(data_dir)

    from sklearn.model_selection import train_test_split
    X = df.drop(columns=['isFraud'])
    y = df['isFraud']

    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    tr_df         = X_tr.copy();  tr_df['isFraud']  = y_tr.values
    val_df        = X_val.copy(); val_df['isFraud']  = y_val.values

    train_eng, freq_maps, les = engineer_features(tr_df,  fit=True)
    val_eng,   _,          _  = engineer_features(
        val_df, freq_maps, les, fit=False
    )

    feature_cols = get_feature_cols(train_eng)

    print(f"\nFeature count  : {len(feature_cols)}")
    print(f"Train shape    : {train_eng[feature_cols].shape}")
    print(f"Val shape      : {val_eng[feature_cols].shape}")
    print(f"Nulls in train : {train_eng[feature_cols].isnull().sum().sum()}")
    print(f"Nulls in val   : {val_eng[feature_cols].isnull().sum().sum()}")

    velocity_cols = [
        'card1_txn_count', 'card1_avg_amt', 'card1_max_amt',
        'card1_fraud_rate', 'amt_vs_card_avg', 'amt_vs_card_max',
        'hour_fraud_rate'
    ]
    print(f"\nVelocity features sample:")
    print(train_eng[velocity_cols].describe().round(3).to_string())