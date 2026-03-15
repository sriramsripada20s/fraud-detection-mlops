"""
schema.py — Pydantic request/response models for the Fraud Detection API

Field selection principle:
  Only fields a real payment system would have at transaction time.
  
  V1-V339, C1-C14, D1-D15 are NOT in the request schema because:
  - These are anonymized vendor features
  - Not available to API callers directly
  - predictor.py fills them with 0 automatically
  - If vendor integration exists, predictor.py can be extended
    to accept them via a separate /predict/enriched endpoint

Available at transaction time:
  - Payment processor : TransactionAmt, TransactionDT, ProductCD
  - Card network      : card1-card6
  - Checkout form     : addr1, addr2, P_emaildomain, R_emaildomain
  - Browser/app SDK   : DeviceType, DeviceInfo, id_30, id_31, id_33
  - Bank/identity     : id_01-id_06
  - Card verification : M4, M6
"""

from pydantic import BaseModel, Field
from typing import Optional


class TransactionRequest(BaseModel):
    """
    Real-time transaction scoring request.
    Only includes fields available at the point of transaction.
    V/C/D vendor features are filled automatically inside predictor.py.
    """

    # ── Required ──────────────────────────────────────────────────
    TransactionAmt: float = Field(
        ..., gt=0,
        description="Transaction amount in USD",
        example=150.0
    )

    # ── Transaction context ───────────────────────────────────────
    TransactionDT: Optional[int] = Field(
        None,
        description="Seconds from reference date",
        example=86400
    )
    ProductCD: Optional[str] = Field(
        None,
        description="Product code: W/H/C/S/R",
        example="W"
    )

    # ── Card info ─────────────────────────────────────────────────
    card1: Optional[int]   = Field(None, description="Card identifier",       example=12345)
    card2: Optional[float] = Field(None, description="Card identifier 2")
    card3: Optional[float] = Field(None, description="Card identifier 3")
    card4: Optional[str]   = Field(None, description="visa/mastercard/amex",  example="visa")
    card5: Optional[float] = Field(None, description="Card identifier 5")
    card6: Optional[str]   = Field(None, description="debit/credit",          example="debit")

    # ── Billing address ───────────────────────────────────────────
    addr1: Optional[float] = Field(None, description="Billing zip code")
    addr2: Optional[float] = Field(None, description="Billing country code")

    # ── Email domains ─────────────────────────────────────────────
    P_emaildomain: Optional[str] = Field(
        None, description="Purchaser email domain", example="gmail.com"
    )
    R_emaildomain: Optional[str] = Field(
        None, description="Recipient email domain"
    )

    # ── Device fingerprint ────────────────────────────────────────
    DeviceType: Optional[str] = Field(
        None, description="mobile or desktop", example="desktop"
    )
    DeviceInfo: Optional[str] = Field(
        None, description="Device model or OS string", example="Windows"
    )
    id_30: Optional[str] = Field(None, description="OS version",         example="Windows 10")
    id_31: Optional[str] = Field(None, description="Browser",            example="chrome 80")
    id_33: Optional[str] = Field(None, description="Screen resolution",  example="1920x1080")

    # ── Identity / bank verification ──────────────────────────────
    id_01: Optional[float] = Field(None, description="Identity score 1")
    id_02: Optional[float] = Field(None, description="Identity score 2")
    id_03: Optional[float] = Field(None, description="Identity score 3")
    id_05: Optional[float] = Field(None, description="Identity score 5")
    id_06: Optional[float] = Field(None, description="Identity score 6")

    # ── Card verification match flags ─────────────────────────────
    M4: Optional[str] = Field(None, description="Match flag T/F", example="T")
    M6: Optional[str] = Field(None, description="Match flag T/F", example="F")

    class Config:
        json_schema_extra = {
            'example': {
                'TransactionAmt': 2500.0,
                'TransactionDT':  3600,
                'ProductCD':      'W',
                'card1':          12345,
                'card4':          'visa',
                'card6':          'debit',
                'P_emaildomain':  'gmail.com',
                'DeviceType':     'desktop',
                'DeviceInfo':     'Windows',
                'id_30':          'Windows 10',
                'id_31':          'chrome 80',
            }
        }


class PredictionResponse(BaseModel):
    """
    Fraud scoring response.

    Action logic:
      score >= threshold       → block
      0.4 <= score < threshold → review (step-up auth / analyst queue)
      score < 0.4              → allow
    """
    fraud_score: float = Field(..., description="Fraud probability [0-1]",    example=0.87)
    is_fraud:    bool  = Field(..., description="True if score >= threshold",  example=True)
    action:      str   = Field(..., description="block | review | allow",      example="block")
    confidence:  str   = Field(..., description="high | medium | low",         example="high")
    threshold:   float = Field(..., description="Threshold used",              example=0.71)


class BatchRequest(BaseModel):
    """Batch scoring — up to 100 transactions per request."""
    transactions: list[TransactionRequest]


class BatchResponse(BaseModel):
    """Batch scoring response."""
    results:     list[PredictionResponse]
    total:       int
    fraud_count: int
    latency_ms:  float


class HealthResponse(BaseModel):
    """API health check — includes model quality metrics."""
    status:        str
    model_loaded:  bool
    threshold:     float
    feature_count: int
    val_ap:        Optional[float] = None
    test_ap:       Optional[float] = None