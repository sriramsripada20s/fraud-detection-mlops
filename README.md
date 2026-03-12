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

## Who This Helps

| Business | Problem Solved |
|---|---|
| Banks & card networks | Block fraudulent transactions before settlement |
| Fintech apps (like Chime, Cash App) | Protect users from unauthorized transfers |
| E-commerce platforms | Detect stolen card usage at checkout |
| Payment processors (Stripe, Square) | Flag suspicious merchant activity |

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
| > threshold | Block | High confidence fraud — reject transaction |
| 0.4 – threshold | Review | Uncertain — flag for analyst or trigger step-up auth (OTP) |
| < 0.4 | Allow | Legitimate — let it through |

## Why Not Just Use Rules?

Traditional fraud systems use hand-written rules like  
*"block if amount > $5000 and country != home country"*.  
These break the moment fraudsters learn the rules.

ML learns complex, non-obvious patterns across 400+ features simultaneously 
and adapts when retrained on new data.

## What We Built
```
Raw transaction data
        ↓
Feature engineering (400+ features, leak-free)
        ↓
XGBoost model (trained on AWS SageMaker)
        ↓
FastAPI serving real-time predictions
        ↓
API Gateway → Lambda → SageMaker endpoint
        ↓
Score + action returned in < 100ms
```

## Business Impact (Expected)

- Catch **65–80% of fraud** while keeping false alarms under 5%
- Sub-100ms response time — no checkout delay for customers
- Automated retraining when fraud patterns drift
- Full audit trail via MLflow experiment tracking

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

## Project Structure
```
fraud-detection-mlops/
├── data/raw/                 # IEEE-CIS CSVs (git-ignored)
├── data/processed/           # EDA outputs, feature groups
├── notebooks/                # EDA — run this first
├── src/                      # preprocess.py, train.py, evaluate.py
├── app/                      # FastAPI — schema, predictor, main
├── tests/                    # Unit + integration tests
├── scripts/                  # AWS deployment scripts
└── .github/workflows/        # CI/CD pipeline
```

## Status
🚧 In active development