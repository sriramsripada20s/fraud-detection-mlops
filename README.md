# Fraud Detection MLOps Pipeline

## Business Problem

Every time a customer swipes a card or makes an online payment, there's a risk 
that transaction is fraudulent. Banks and fintech companies lose billions every 
year to fraud — but the bigger challenge is catching it **in real time**, before 
the money moves.

The problem is hard because:
- Fraud is rare — only ~3.5% of transactions are fraudulent (heavily imbalanced)
- Fraudsters constantly change their patterns
- A wrong block frustrates a legitimate customer and hurts the business
- Decisions need to happen in milliseconds, not minutes

---

## Who This Helps

| Business | Problem Solved |
|---|---|
| Banks & card networks | Block fraudulent transactions before settlement |
| Fintech apps (like Chime, Cash App) | Protect users from unauthorized transfers |
| E-commerce platforms | Detect stolen card usage at checkout |
| Payment processors (Stripe, Square) | Flag suspicious merchant activity |

---

## How We Solve It

We train a machine learning model on 590,000 real transactions from the 
IEEE-CIS Fraud Detection dataset (Kaggle). The model learns patterns that 
separate fraudulent transactions from legitimate ones — things like:

- Unusual transaction amounts at odd hours
- Mismatched email domains between sender and receiver
- Cards being used at abnormal frequencies
- Missing device/identity information (fraudsters avoid leaving traces)

The model outputs a **fraud probability score** between 0 and 1 for every 
transaction. Based on that score, we take one of three actions:

| Score | Action | Meaning |
|---|---|---|
| >= threshold | Block | High confidence fraud — reject transaction |
| 0.4 – threshold | Review | Uncertain — flag for analyst or trigger step-up auth (OTP) |
| < 0.4 | Allow | Legitimate — let it through |

---

## Why Not Just Use Rules?

Traditional fraud systems use hand-written rules like  
*"block if amount > $5000 and country != home country"*.  
These break the moment fraudsters learn the rules.

ML learns complex, non-obvious patterns across 400+ features simultaneously 
and adapts when retrained on new data.

---

## Two Model Approach

During development we discovered that anonymized features (V1-V339, C1-C14,
D1-D15) provided by a third-party data vendor dominate model importance but
their real-time availability depends on an external API integration.
We quantified this gap honestly and built two models:

| Model | Features Used | Test AP | Fraud Caught | Use Case |
|---|---|---|---|---|
| **Full model** | All features incl vendor features | 0.7162 | ~72% | With third-party data vendor integration |
| **Real-time model** | Real-time available features only | 0.5657 | ~58% | Standalone, no vendor dependency |

The AP drop of **0.15** from removing vendor features means roughly 14 in 100
fraud cases are missed without third-party integration. This trade-off is
documented honestly rather than hidden.

**Real-time available features used in standalone model:**
- Transaction: amount, product code, time-of-day
- Card: card network, type, identifier frequency
- Address: billing zip, country frequency
- Device: device type, OS, browser, screen resolution
- Identity: identity scores id_01 to id_06
- Engineered: card velocity, fraud rate per card, amount anomaly ratio

---

## Model Performance

### Baseline (minimal preprocessing, notebook)
| Metric | Value |
|---|---|
| Avg Precision | 0.6257 |
| Recall (fraud) | 0.6747 |
| Fraud caught | 67.47% |
| False alarm rate | 3.65% |

### Production model (real-time features + velocity engineering)
| Metric | Value |
|---|---|
| Avg Precision | **0.5657** |
| AUC-ROC | 0.9518 |
| Threshold (F2 tuned) | 0.6999 |
| Overfit gap (train - val AP) | 0.08 |

> Full model with all features achieves AP=0.7162 but requires third-party vendor integration.

---

## Key Engineering Decisions

Every decision is backed by EDA notebooks:

| Decision | EDA Evidence |
|---|---|
| `scale_pos_weight=28` | 3.5% fraud rate = 28:1 class ratio |
| Average Precision as primary metric | AUC-ROC misleading on 3.5% fraud |
| F2 threshold tuning (beta=2) | Recall more important than precision in fraud |
| Missing identity flag | Missing identity = 2x fraud rate |
| Frequency encode card1/addr | High cardinality, velocity signal |
| log1p amount transform | Right-skewed distribution |
| is_night binary feature | Night hours show elevated fraud rate |
| card1_fraud_rate | Historical fraud rate per card — built from transaction history |
| amt_vs_card_avg | Amount anomaly vs card's typical behaviour |
| Exclude V/C/D columns | Unknown real-time availability, replaced with engineered features |

---

## What We Built

```
Raw transaction data
        ↓
Feature engineering (real-time available features + velocity features, leak-free)
        ↓
XGBoost model (trained locally + AWS SageMaker)
        ↓
FastAPI serving real-time predictions
        ↓
API Gateway → Lambda → SageMaker endpoint
        ↓
Score + action returned in < 100ms
```

---

## Tech Stack

| Layer | Tool |
|---|---|
| ML model | XGBoost |
| Experiment tracking | MLflow |
| API | FastAPI |
| Containerisation | Docker |
| CI/CD | GitHub Actions |
| Cloud training | AWS SageMaker |
| Storage | AWS S3 |
| Serving | SageMaker endpoint → Lambda → API Gateway |

---

## Project Structure

```
fraud-detection-mlops/
├── data/raw/                 # IEEE-CIS CSVs (git-ignored)
├── data/processed/           # EDA outputs, null profile, metrics
├── notebooks/
│   ├── 01_eda_fraud_detection.ipynb     # Class imbalance, amounts, time
│   ├── 02_eda_categorical_missing.ipynb # Categoricals, missing values
│   ├── 03_eda_vcols_identity.ipynb      # V-columns, identity block
│   └── 04_baseline_model.ipynb         # Baseline XGBoost AP=0.6257
├── src/
│   ├── preprocess.py         # Feature engineering (leak-free, no vendor features)
│   ├── train.py              # Training pipeline + MLflow tracking
│   └── evaluate.py           # Threshold tuning + metrics
├── app/
│   ├── schema.py             # Pydantic request/response models
│   ├── predictor.py          # Model loader + inference
│   └── main.py               # FastAPI application
├── tests/
│   ├── test_preprocess.py    # Preprocessing unit tests
│   └── test_api.py           # API integration tests
├── scripts/
│   ├── upload_data.py        # Upload data to S3
│   ├── sagemaker_train.py    # Launch SageMaker training job
│   └── sagemaker_deploy.py   # Deploy SageMaker endpoint
└── .github/workflows/
    └── deploy.yml            # CI/CD pipeline
```

---

## Quickstart (local)

```bash
# 1. Clone and install
git clone https://github.com/sriramsripada20s/fraud-detection-mlops
cd fraud-detection-mlops
python -m venv fraudenv
fraudenv\Scripts\activate
pip install -r requirements.txt

# 2. Download data
kaggle competitions download -c ieee-fraud-detection -p data/raw

# 3. Run EDA notebooks
jupyter notebook notebooks/

# 4. Start MLflow (Terminal 1)
mlflow ui --port 5000

# 5. Train model (Terminal 2)
cd src && python train.py

# 6. Start API
cd app && uvicorn main:app --reload --port 8000

# 7. Test
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"TransactionAmt": 2500.0, "card4": "visa", "card6": "debit"}'
```

---

## API Endpoints

```
GET  /health           — model status + metrics
POST /predict          — single transaction scoring
POST /predict/batch    — up to 100 transactions
GET  /docs             — Swagger UI
```

---

## AWS Deployment

```bash
# Upload data to S3
python scripts/upload_data.py

# Launch SageMaker training job
python scripts/sagemaker_train.py

# Deploy endpoint
python scripts/sagemaker_deploy.py

# Delete endpoint when done (saves cost ~$0.056/hr)
python scripts/sagemaker_deploy.py delete
```

GitHub secrets required: `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `SAGEMAKER_ROLE_ARN`

---

## Status

✅ EDA complete (4 notebooks)
✅ Preprocessing pipeline (leak-free, velocity features, no vendor dependency)
✅ Training pipeline (MLflow tracking, 3-way split, baseline comparison)
✅ Two model approach documented honestly (AP=0.7162 vs AP=0.5657)
🚧 FastAPI serving
🚧 Tests
🚧 Docker + CI/CD
🚧 AWS SageMaker deployment