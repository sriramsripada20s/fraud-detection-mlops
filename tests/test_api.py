"""
test_api.py — Critical API integration tests

Only testing what matters most:
  1. Health endpoint returns model metadata
  2. Predict returns valid response schema
  3. Predict score is within valid range [0-1]
  4. Batch predict returns correct count
  5. Missing required field returns 422
  6. Batch over limit returns 400

Run:
  pytest tests/test_api.py -v

Note: model must exist at model/model.joblib before running.
      Run src/train.py first.
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'app'))

from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

# ------------------------------------------------------------------
# Test payloads
# ------------------------------------------------------------------
NORMAL_TXN = {
    'TransactionAmt': 49.99,
    'card4':          'visa',
    'card6':          'debit',
    'P_emaildomain':  'gmail.com',
    'DeviceType':     'desktop',
}

HIGH_RISK_TXN = {
    'TransactionAmt': 9999.0,
    'TransactionDT':  3600,
    'card4':          'mastercard',
    'card6':          'credit',
    'P_emaildomain':  'anonymous.com',
    'DeviceType':     'mobile',
}


# ------------------------------------------------------------------
# Test 1 — Health endpoint returns model metadata
# ------------------------------------------------------------------
def test_health_returns_model_metadata():
    r    = client.get('/health')
    data = r.json()

    assert r.status_code          == 200
    assert data['status']         == 'healthy'
    assert data['model_loaded']   == True
    assert isinstance(data['threshold'],     float)
    assert isinstance(data['feature_count'], int)
    assert data['feature_count']  > 50


# ------------------------------------------------------------------
# Test 2 — Predict returns valid response schema
# ------------------------------------------------------------------
def test_predict_returns_valid_schema():
    r    = client.post('/predict', json=NORMAL_TXN)
    data = r.json()

    assert r.status_code   == 200
    assert 'fraud_score'   in data
    assert 'is_fraud'      in data
    assert 'action'        in data
    assert 'confidence'    in data
    assert 'threshold'     in data
    assert data['action']     in ('block', 'review', 'allow')
    assert data['confidence'] in ('high', 'medium', 'low')


# ------------------------------------------------------------------
# Test 3 — Fraud score is within valid range [0-1]
# ------------------------------------------------------------------
def test_predict_score_range():
    for txn in [NORMAL_TXN, HIGH_RISK_TXN]:
        r    = client.post('/predict', json=txn)
        data = r.json()
        assert 0.0 <= data['fraud_score'] <= 1.0, \
            f"Score out of range: {data['fraud_score']}"


# ------------------------------------------------------------------
# Test 4 — Batch predict returns correct count
# ------------------------------------------------------------------
def test_batch_predict_returns_correct_count():
    txns = [NORMAL_TXN, HIGH_RISK_TXN, NORMAL_TXN]
    r    = client.post('/predict/batch', json={'transactions': txns})
    data = r.json()

    assert r.status_code       == 200
    assert data['total']       == 3
    assert len(data['results']) == 3
    assert 'fraud_count'       in data
    assert 'latency_ms'        in data


# ------------------------------------------------------------------
# Test 5 — Missing required field returns 422
# ------------------------------------------------------------------
def test_missing_required_field_returns_422():
    # TransactionAmt is required — missing it should fail validation
    r = client.post('/predict', json={'card4': 'visa'})
    assert r.status_code == 422


# ------------------------------------------------------------------
# Test 6 — Batch over 100 returns 400
# ------------------------------------------------------------------
def test_batch_over_limit_returns_400():
    txns = [NORMAL_TXN] * 101
    r    = client.post('/predict/batch', json={'transactions': txns})
    assert r.status_code == 400