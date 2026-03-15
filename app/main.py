"""
main.py — FastAPI fraud detection API

Endpoints:
  GET  /health           — liveness check + model metadata
  POST /predict          — single transaction scoring
  POST /predict/batch    — up to 100 transactions
  GET  /docs             — Swagger UI (auto-generated)

Run:
  cd app
  uvicorn main:app --reload --port 8000

Test:
  curl http://localhost:8000/health
  curl -X POST http://localhost:8000/predict \
    -H "Content-Type: application/json" \
    -d '{"TransactionAmt": 2500.0, "card4": "visa", "card6": "debit"}'
"""

import time
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from app.schema import (
    TransactionRequest, PredictionResponse,
    BatchRequest, BatchResponse, HealthResponse,
)
from app.predictor import get_predictor, FraudPredictor

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
)
logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Lifespan — load model once at startup
# ------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up — loading model...")
    get_predictor()
    logger.info("Model ready.")
    yield
    logger.info("Shutting down.")


# ------------------------------------------------------------------
# App
# ------------------------------------------------------------------
app = FastAPI(
    title='Fraud Detection API',
    description=(
        'Real-time transaction fraud scoring.\n\n'
        'IEEE-CIS dataset · XGBoost · F2 threshold-optimised.\n\n'
        '**Performance:**\n'
        '- Test AP: 0.7162 (with vendor features)\n'
        '- Test AP: 0.5484 (without vendor features)\n'
        '- Fraud caught: 73.44%\n'
        '- False alarm rate: 2.23%'
    ),
    version='1.0.0',
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_methods=['*'],
    allow_headers=['*'],
)


# ------------------------------------------------------------------
# Request timing middleware
# ------------------------------------------------------------------
@app.middleware('http')
async def add_process_time(request: Request, call_next):
    start    = time.time()
    response = await call_next(request)
    ms       = round((time.time() - start) * 1000, 2)
    response.headers['X-Process-Time-Ms'] = str(ms)
    return response


# ------------------------------------------------------------------
# Routes
# ------------------------------------------------------------------
@app.get('/', tags=['ops'])
def root():
    return {
        'service': 'Fraud Detection API',
        'version': '1.0.0',
        'docs':    '/docs',
        'health':  '/health',
    }


@app.get('/health', response_model=HealthResponse, tags=['ops'])
def health(p: FraudPredictor = Depends(get_predictor)):
    """
    Model health check.
    Returns model status, threshold, feature count and val/test metrics.
    """
    return {
        'status':        'healthy',
        'model_loaded':  True,
        'threshold':     p.threshold,
        'feature_count': len(p.feature_cols),
        'val_ap':        p.val_metrics.get('avg_precision'),
        'test_ap':       None,  # loaded from test_results.json if needed
    }


@app.post('/predict', response_model=PredictionResponse, tags=['inference'])
def predict(
    request: TransactionRequest,
    p: FraudPredictor = Depends(get_predictor),
):
    """
    Score a single transaction.

    Only TransactionAmt is required.
    All other fields are optional — missing fields use safe defaults.
    V/C/D vendor features are filled with 0 automatically.

    Returns fraud_score, is_fraud, action (block/review/allow), confidence.
    """
    start = time.time()
    try:
        payload = request.model_dump()
        result  = p.predict(payload)
        ms      = round((time.time() - start) * 1000, 2)

        logger.info(
            f"predict | amt={payload.get('TransactionAmt')} "
            f"| score={result['fraud_score']} "
            f"| action={result['action']} "
            f"| {ms}ms"
        )

        if result['is_fraud']:
            logger.warning(
                f"FRAUD DETECTED | score={result['fraud_score']} "
                f"| amt={payload.get('TransactionAmt')} "
                f"| card4={payload.get('card4')} "
                f"| device={payload.get('DeviceType')}"
            )

        return result

    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post('/predict/batch', response_model=BatchResponse, tags=['inference'])
def predict_batch(
    request: BatchRequest,
    p: FraudPredictor = Depends(get_predictor),
):
    """
    Score up to 100 transactions in a single request.

    Returns results list, total count, fraud count and latency.
    """
    if len(request.transactions) > 100:
        raise HTTPException(
            status_code=400,
            detail='Maximum 100 transactions per batch request.'
        )

    start   = time.time()
    results = p.predict_batch(
        [t.model_dump() for t in request.transactions]
    )
    ms      = round((time.time() - start) * 1000, 2)

    fraud_count = sum(1 for r in results if r['is_fraud'])

    logger.info(
        f"batch_predict | n={len(results)} "
        f"| fraud={fraud_count} "
        f"| {ms}ms"
    )

    return {
        'results':     results,
        'total':       len(results),
        'fraud_count': fraud_count,
        'latency_ms':  ms,
    }