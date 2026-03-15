"""
preprocess.py — IEEE-CIS Fraud Detection feature engineering

Every decision here is backed by EDA notebooks:
  - V-cols fill 0          → notebooks/03 (null groups analysis)
  - id numeric fill -1     → notebooks/03 (missingness = fraud signal)
  - freq encoding leak-free→ notebooks/02 (high cardinality analysis)
  - time features          → notebooks/02 (night uplift analysis)
  - amount features        → notebooks/02 (log1p, decimal signal)

Usage:
  from preprocess import load_data, engineer_features, get_feature_cols

EDA Finding → preprocess.py implementation

From Notebook 02 (amounts + time):
EDA: TransactionAmt is right-skewed
  → amt_log = log1p(TransactionAmt)

EDA: Decimal part of amount differs between fraud/legit
  → amt_decimal = TransactionAmt % 1
  → amt_is_round = binary flag

EDA: Night hours have elevated fraud rate
  → is_night = (hour >= 22 or hour <= 5)
  → is_weekend = binary flag

From Notebook 03 (categoricals + missing values):
EDA: Missing identity = 2x fraud rate
  → has_identity = binary flag
  → id_01_missing, id_02_missing... per column flags

EDA: M-flag null ≠ False (different fraud rates)
  → fill 'unknown' not NaN before label encoding

EDA: card1/addr1 high cardinality
  → frequency encode — count how often each card appears

From Notebook 04 (V-columns):
EDA: V-col nulls appear in groups (feature blocks)
EDA: Null means feature not triggered
  → fill 0 not -999
  → -999 misleads XGBoost as extreme outlier
"""



import os
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import joblib
from typing import Optional

# ------------------------------------------------------------------
# Column group definitions — validated against feature_groups.json
# ------------------------------------------------------------------
CAT_COLS = [
    'ProductCD', 'card4', 'card6',
    'P_emaildomain', 'R_emaildomain',
    'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9',
]

ID_CAT_COLS = [
    'id_12', 'id_15', 'id_16', 'id_23', 'id_27', 'id_28',
    'id_29', 'id_30', 'id_31', 'id_32', 'id_33', 'id_34',
    'id_35', 'id_36', 'id_37', 'id_38',
]

# High-cardinality → frequency encode (computed on train only)
FREQ_COLS    = ['card1', 'card2', 'addr1', 'addr2']

# Vesta feature columns
V_COLS       = [f'V{i}' for i in range(1, 340)]

# Numeric identity block
ID_NUM_COLS  = [f'id_{str(i).zfill(2)}' for i in range(1, 12)]

# Drop after feature extraction
DROP_COLS    = ['TransactionID', 'TransactionDT']


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

    import gc
    del txn, idn
    gc.collect()

    print(f"Loaded   : {df.shape[0]:,} rows x {df.shape[1]} cols")
    print(f"Fraud    : {df['isFraud'].mean()*100:.2f}%")
    print(f"Memory   : {df.memory_usage(deep=True).sum()/1e6:.0f} MB")
    return df


def load_test_data(data_dir: str) -> pd.DataFrame:
    """Load and merge test tables (no isFraud column)."""
    v_dtypes   = {f'V{i}': 'float32' for i in range(1, 340)}
    txn = pd.read_csv(
        os.path.join(data_dir, 'test_transaction.csv'),
        dtype=v_dtypes
    )
    idn = pd.read_csv(os.path.join(data_dir, 'test_identity.csv'))
    df  = txn.merge(idn, on='TransactionID', how='left')
    import gc; del txn, idn; gc.collect()
    return df


# ------------------------------------------------------------------
# Core feature engineering
# ------------------------------------------------------------------
def engineer_features(
    df: pd.DataFrame,
    freq_maps: Optional[dict]  = None,
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

    if freq_maps     is None: freq_maps     = {}
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
    #    EDA finding: card1/addr1 high cardinality, velocity signal
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
    # 4. Identity presence flag
    #    EDA finding: missing identity = 2x fraud rate
    # ----------------------------------------------------------
    df['has_identity'] = df['id_01'].notnull().astype(int) \
                         if 'id_01' in df.columns else 0

    # ----------------------------------------------------------
    # 5. Missingness flags for numeric identity block (id_01..id_11)
    #    EDA finding: null is NOT random — fraudsters skip identity
    # ----------------------------------------------------------
    for col in ID_NUM_COLS:
        if col in df.columns:
            df[f'{col}_missing'] = df[col].isnull().astype(int)

    # ----------------------------------------------------------
    # 6. Null filling by column type
    # ----------------------------------------------------------

    # V-columns → fill 0
    # EDA finding: null = feature not triggered, 0 is correct
    # -999 would mislead XGBoost as extreme outlier
    v_present = [c for c in V_COLS if c in df.columns]
    df[v_present] = df[v_present].fillna(0)

    # id numeric → fill -1 (distinct from 0 = present but zero)
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
    #    EDA finding: label encode sufficient for tree models
    #    fill 'unknown' preserves null as a distinct category
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
                known    = set(le.classes_)
                df[col]  = df[col].apply(
                    lambda x: x if x in known else 'unknown'
                )
                df[col] = le.transform(df[col])
            else:
                df[col] = 0

    # ----------------------------------------------------------
    # 8. Fill any remaining nulls with 0
    # ----------------------------------------------------------
    df = df.fillna(0)

    # ----------------------------------------------------------
    # 9. Drop columns not needed for training
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
# CLI entry point — quick validation
# ------------------------------------------------------------------
if __name__ == '__main__':
    data_dir = sys.argv[1] if len(sys.argv) > 1 else '../data/raw'

    df = load_data(data_dir)

    # Split first, then engineer — prevents leakage
    from sklearn.model_selection import train_test_split
    X = df.drop(columns=['isFraud'])
    y = df['isFraud']

    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    tr_df        = X_tr.copy(); tr_df['isFraud']  = y_tr.values
    val_df       = X_val.copy(); val_df['isFraud'] = y_val.values

    train_eng, freq_maps, les = engineer_features(tr_df,  fit=True)
    val_eng,   _,          _  = engineer_features(
        val_df, freq_maps, les, fit=False
    )

    feature_cols = get_feature_cols(train_eng)

    print(f"\nFeature count : {len(feature_cols)}")
    print(f"Train shape   : {train_eng[feature_cols].shape}")
    print(f"Val shape     : {val_eng[feature_cols].shape}")
    print(f"Nulls in train: {train_eng[feature_cols].isnull().sum().sum()}")
    print(f"Nulls in val  : {val_eng[feature_cols].isnull().sum().sum()}")

    os.makedirs('../data/processed', exist_ok=True)
    save_preprocessing_artifacts(
        freq_maps, les, feature_cols,
        path='../model/preprocessing.joblib'
    )
