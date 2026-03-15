Fraud Detection MLOps Pipeline
Business Problem
Every time a customer swipes a card or makes an online payment, there's a risk
that transaction is fraudulent. Banks and fintech companies lose billions every
year to fraud — but the bigger challenge is catching it in real time, before
the money moves.
The problem is hard because:

Fraud is rare — only ~3.5% of transactions are fraudulent (heavily imbalanced)
Fraudsters constantly change their patterns
A wrong block frustrates a legitimate customer and hurts the business
Decisions need to happen in milliseconds, not minutes


Who This Helps
BusinessProblem SolvedBanks & card networksBlock fraudulent transactions before settlementFintech apps (like Chime, Cash App)Protect users from unauthorized transfersE-commerce platformsDetect stolen card usage at checkoutPayment processors (Stripe, Square)Flag suspicious merchant activity

How We Solve It
We train a machine learning model on 590,000 real transactions from the
IEEE-CIS Fraud Detection dataset (Kaggle). The model learns patterns that
separate fraudulent transactions from legitimate ones — things like:

Unusual transaction amounts at odd hours
Mismatched email domains between sender and receiver
Cards being used at abnormal frequencies
Missing device/identity information (fraudsters avoid leaving traces)

The model outputs a fraud probability score between 0 and 1 for every
transaction. Based on that score, we take one of three actions:
ScoreActionMeaning>= thresholdBlockHigh confidence fraud — reject transaction0.4 – thresholdReviewUncertain — flag for analyst or trigger step-up auth (OTP)< 0.4AllowLegitimate — let it through

Why Not Just Use Rules?
Traditional fraud systems use hand-written rules like
"block if amount > $5000 and country != home country".
These break the moment fraudsters learn the rules.
ML learns complex, non-obvious patterns across 400+ features simultaneously
and adapts when retrained on new data.

Feature Strategy
The model is trained on all available features including anonymized vendor
features (V1-V339, C1-C14, D1-D15). The API exposes only understandable,
real-time available fields to the caller.
LayerFeaturesAPI requestTransactionAmt, card info, device, email, address, identityModel trainingAll 456 features including vendor featuresInference without vendorVendor features default to 0 automatically
Performance impact of vendor features:
ScenarioTest APFraud CaughtWith vendor features (V/C/D provided)0.716272%Without vendor features (V/C/D = 0)0.548467%
This trade-off is documented honestly. In production, integrating a third-party
fraud data vendor would close this gap.

Model Performance
Baseline (minimal preprocessing, notebook)
MetricValueAvg Precision0.6257Recall (fraud)0.6747Fraud caught67.47%False alarm rate3.65%
Production model (full features + velocity engineering)
MetricValueTest Avg Precision0.7097Test Recall (fraud)0.7218Fraud caught72.18%False alarm rate2.19%AUC-ROC0.9518Threshold (F2 tuned)0.6999Overfit gap (train - val AP)0.08

Key Engineering Decisions
Every decision is backed by EDA notebooks:
DecisionEDA Evidencescale_pos_weight=283.5% fraud rate = 28:1 class ratioAverage Precision as primary metricAUC-ROC misleading on 3.5% fraudF2 threshold tuning (beta=2)Recall more important than precision in fraudMissing identity flagMissing identity = 2x fraud rateFrequency encode card1/addrHigh cardinality, velocity signallog1p amount transformRight-skewed distributionis_night binary featureNight hours show elevated fraud ratecard1_fraud_rateHistorical fraud rate per card — built from transaction historyamt_vs_card_avgAmount anomaly vs card's typical behaviourV/C/D cols fill 0Null = feature not triggered, not unknown value

What We Built
Raw transaction data
        ↓
Feature engineering (456 features, leak-free)
        ↓
XGBoost model (trained locally + AWS SageMaker)
        ↓
FastAPI — accepts real-time understandable fields
        ↓
predictor.py fills V/C/D = 0 if vendor not integrated
        ↓
API Gateway → Lambda → SageMaker endpoint
        ↓
Score + action returned in < 100ms

Tech Stack
LayerToolML modelXGBoostExperiment trackingMLflowAPIFastAPIContainerisationDockerCI/CDGitHub ActionsCloud trainingAWS SageMakerStorageAWS S3ServingSageMaker endpoint → Lambda → API Gateway

Project Structure
fraud-detection-mlops/
├── data/raw/                 # IEEE-CIS CSVs (git-ignored)
├── data/processed/           # EDA outputs, null profile, metrics
├── notebooks/
│   ├── 01_eda_fraud_detection.ipynb     # Class imbalance, amounts, time
│   ├── 02_eda_categorical_missing.ipynb # Categoricals, missing values
│   ├── 03_eda_vcols_identity.ipynb      # V-columns, identity block
│   └── 04_baseline_model.ipynb         # Baseline XGBoost AP=0.6257
├── src/
│   ├── preprocess.py         # Feature engineering (leak-free)
│   ├── train.py              # Training pipeline + MLflow tracking
│   └── evaluate.py           # Threshold tuning + metrics
├── app/
│   ├── schema.py             # Pydantic — real-time fields only
│   ├── predictor.py          # Model loader + inference (fills V/C/D=0)
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

Quickstart (local)
bash# 1. Clone and install
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

API Endpoints
GET  /health           — model status + metrics
POST /predict          — single transaction scoring
POST /predict/batch    — up to 100 transactions
GET  /docs             — Swagger UI

AWS Deployment
bash# Upload data to S3
python scripts/upload_data.py

# Launch SageMaker training job
python scripts/sagemaker_train.py

# Deploy endpoint
python scripts/sagemaker_deploy.py

# Delete endpoint when done (saves cost ~$0.056/hr)
python scripts/sagemaker_deploy.py delete
GitHub secrets required: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, SAGEMAKER_ROLE_ARN

Status
✅ EDA complete (4 notebooks)
✅ Preprocessing pipeline (leak-free, velocity features)
✅ Training pipeline (MLflow tracking, 3-way split, baseline comparison)
✅ Vendor feature strategy documented (AP=0.7162 with / AP=0.5484 without)
🚧 FastAPI serving
🚧 Tests
🚧 Docker + CI/CD
🚧 AWS SageMaker deployment