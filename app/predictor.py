"""
predictor.py — Model loader and inference engine

Responsibilities:
  1. Load model.joblib once at startup (singleton pattern)
  2. Accept 22 real-time fields from API request
  3. Build full 462-feature row — fills V/C/D = 0 if not provided
  4. Apply same freq_maps and label_encoders used during training
  5. Return fraud score, action, confidence

V/C/D vendor features:
  - Default to 0 at inference if vendor not integrated
  - AP with vendor (V/C/D provided) : 0.7162
  - AP without vendor (V/C/D = 0)   : 0.5484
  - Documented in README

Singleton pattern:
  Model loads once at API startup via get_predictor()
  Not reloaded per request — keeps latency low
"""

import os
import io
import logging
import numpy as np
import pandas as pd
import joblib
from typing import Optional

logger = logging.getLogger(__name__)

MODEL_PATH   = os.getenv('MODEL_PATH',   '../model/model.joblib')
MODEL_BUCKET = os.getenv('MODEL_BUCKET', '')
MODEL_KEY    = os.getenv('MODEL_KEY',    'model/model.joblib')


class FraudPredictor:

    def __init__(self):
        self.model          = None
        self.feature_cols   = None
        self.threshold      = 0.5
        self.freq_maps      = {}
        self.label_encoders = {}
        self.val_metrics    = {}
        self._load()

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    def _load(self):
        path = self._resolve_path()
        logger.info(f"Loading model from {path}")

        artifact = joblib.load(path)

        self.model          = artifact['model']
        self.feature_cols   = artifact['feature_cols']
        self.threshold      = artifact.get('threshold', 0.5)
        self.freq_maps      = artifact.get('freq_maps', {})
        self.label_encoders = artifact.get('label_encoders', {})
        self.val_metrics    = artifact.get('val_metrics', {})

        logger.info(
            f"Model loaded | features={len(self.feature_cols)} "
            f"| threshold={self.threshold:.4f} "
            f"| val_ap={self.val_metrics.get('avg_precision', 'n/a')}"
        )

    def _resolve_path(self) -> str:
        """Try local path first — fall back to S3 if MODEL_BUCKET set."""
        if os.path.exists(MODEL_PATH):
            return MODEL_PATH
        if MODEL_BUCKET:
            return self._download_from_s3()
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. "
            f"Run src/train.py first or set MODEL_BUCKET env var."
        )

    def _download_from_s3(self) -> str:
        import boto3
        logger.info(f"Downloading model from s3://{MODEL_BUCKET}/{MODEL_KEY}")
        s3  = boto3.client('s3')
        buf = io.BytesIO()
        s3.download_fileobj(MODEL_BUCKET, MODEL_KEY, buf)
        buf.seek(0)
        return buf

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def predict(self, payload: dict) -> dict:
        """
        Score a single transaction.

        Parameters
        ----------
        payload : dict from TransactionRequest.model_dump()

        Returns
        -------
        dict matching PredictionResponse schema
        """
        row = self._build_feature_row(payload)
        X   = pd.DataFrame([row])[self.feature_cols]

        score    = float(self.model.predict_proba(X)[0][1])
        is_fraud = score >= self.threshold

        # Confidence bands
        if score >= 0.85:
            confidence = 'high'
        elif score >= 0.60:
            confidence = 'medium'
        else:
            confidence = 'low'

        # Action logic
        if is_fraud:
            action = 'block'
        elif score >= 0.40:
            action = 'review'
        else:
            action = 'allow'

        return {
            'fraud_score': round(score, 4),
            'is_fraud':    is_fraud,
            'action':      action,
            'confidence':  confidence,
            'threshold':   self.threshold,
        }

    def predict_batch(self, payloads: list) -> list:
        """Score multiple transactions."""
        return [self.predict(p) for p in payloads]

    # ------------------------------------------------------------------
    # Feature row builder
    # ------------------------------------------------------------------
    def _build_feature_row(self, payload: dict) -> dict:
        """
        Build a full 462-feature row from API payload.

        API provides 22 real-time fields.
        Everything else is filled using training artifacts or defaults.

        V/C/D vendor features default to 0.
        id columns not in payload default to -1.
        Categoricals not in payload default to 'unknown'.
        """
        row = {}

        amt = float(payload.get('TransactionAmt', 0) or 0)
        dt  = int(payload.get('TransactionDT', 86400) or 86400)

        # ── Time features ─────────────────────────────────────────
        hour            = (dt // 3600) % 24
        row['hour']        = hour
        row['day_of_week'] = (dt // 86400) % 7
        row['is_night']    = int(hour >= 22 or hour <= 5)
        row['is_weekend']  = int(row['day_of_week'] >= 5)

        # ── Amount features ───────────────────────────────────────
        row['TransactionAmt'] = amt
        row['amt_log']        = np.log1p(amt)
        row['amt_decimal']    = amt % 1
        row['amt_cents']      = int(amt * 100) % 100
        row['amt_is_round']   = int((amt % 1) == 0)

        # ── Frequency encoding ────────────────────────────────────
        for col in ['card1', 'card2', 'addr1', 'addr2']:
            freq_map      = self.freq_maps.get(col, {})
            raw_val       = payload.get(col)
            row[f'{col}_freq'] = int(freq_map.get(raw_val, 0))

        # ── Velocity features ─────────────────────────────────────
        card1_val      = payload.get('card1')
        card_stats_map = self.freq_maps.get('card1_stats', {})
        card_stats     = card_stats_map.get(card1_val, {})

        row['card1_txn_count']  = card_stats.get('card1_txn_count',  0)
        row['card1_avg_amt']    = card_stats.get('card1_avg_amt',    0)
        row['card1_max_amt']    = card_stats.get('card1_max_amt',    0)
        row['card1_fraud_rate'] = card_stats.get('card1_fraud_rate', 0)

        card_avg = row['card1_avg_amt']
        card_max = row['card1_max_amt']
        row['amt_vs_card_avg'] = amt / card_avg if card_avg > 0 else 1.0
        row['amt_vs_card_max'] = amt / card_max if card_max > 0 else 1.0

        hour_map = self.freq_maps.get('hour_fraud_rate', {})
        row['hour_fraud_rate'] = float(hour_map.get(hour, 0))

        # ── Identity presence ─────────────────────────────────────
        row['has_identity'] = int(payload.get('id_01') is not None)

        # ── Numeric identity block (id_01..id_11) ─────────────────
        id_num_cols = [f'id_{str(i).zfill(2)}' for i in range(1, 12)]
        for col in id_num_cols:
            val         = payload.get(col)
            row[col]    = float(val) if val is not None else -1.0
            row[f'{col}_missing'] = int(val is None)

        # ── Card / addr numerics ──────────────────────────────────
        for col in ['card1', 'card2', 'card3', 'card5']:
            val      = payload.get(col)
            median   = self.freq_maps.get(f'{col}_median', 0)
            row[col] = float(val) if val is not None else median

        for col in ['addr1', 'addr2', 'dist1', 'dist2']:
            val      = payload.get(col)
            row[col] = float(val) if val is not None else 0.0

        # ── Categorical encoding ──────────────────────────────────
        cat_cols = [
            'ProductCD', 'card4', 'card6',
            'P_emaildomain', 'R_emaildomain',
            'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9',
            'DeviceType', 'DeviceInfo',
            'id_12', 'id_15', 'id_16', 'id_23', 'id_27', 'id_28',
            'id_29', 'id_30', 'id_31', 'id_32', 'id_33', 'id_34',
            'id_35', 'id_36', 'id_37', 'id_38',
        ]
        for col in cat_cols:
            le  = self.label_encoders.get(col)
            raw = str(payload.get(col) or 'unknown')
            if le is not None:
                if raw not in set(le.classes_):
                    raw = 'unknown'
                row[col] = int(le.transform([raw])[0])
            else:
                row[col] = 0

        # ── V-columns — vendor features default to 0 ─────────────
        # AP with vendor (provided): 0.7162
        # AP without vendor (zeros): 0.5484
        for i in range(1, 340):
            row[f'V{i}'] = float(payload.get(f'V{i}', 0) or 0)

        # ── C-columns — vendor count features default to 0 ────────
        for i in range(1, 15):
            row[f'C{i}'] = float(payload.get(f'C{i}', 0) or 0)

        # ── D-columns — vendor time delta features default to 0 ───
        for i in range(1, 16):
            row[f'D{i}'] = float(payload.get(f'D{i}', 0) or 0)

        # ── Fill any remaining feature cols with 0 ────────────────
        for col in self.feature_cols:
            if col not in row:
                row[col] = 0

        return row


# ------------------------------------------------------------------
# Module-level singleton
# ------------------------------------------------------------------
_predictor: Optional[FraudPredictor] = None


def get_predictor() -> FraudPredictor:
    """Return the singleton predictor — loads model on first call."""
    global _predictor
    if _predictor is None:
        _predictor = FraudPredictor()
    return _predictor