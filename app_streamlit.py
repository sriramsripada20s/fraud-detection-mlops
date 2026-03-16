"""
app_streamlit.py — Fraud Detection UI

Calls the live API Gateway endpoint and displays results.

Run locally:
  streamlit run app_streamlit.py

Deploy:
  Push to GitHub → connect on streamlit.io → auto deploys
"""

import streamlit as st
import requests
import json

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------
API_URL = ""

st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="🔍",
    layout="centered",
)

# ------------------------------------------------------------------
# Header
# ------------------------------------------------------------------
st.title("🔍 Fraud Detection System")
st.markdown(
    "Real-time transaction fraud scoring powered by XGBoost + AWS SageMaker."
)
st.markdown("---")

# ------------------------------------------------------------------
# Input form
# ------------------------------------------------------------------
st.subheader("Transaction Details")

col1, col2 = st.columns(2)

with col1:
    amount = st.number_input(
        "Transaction Amount ($)",
        min_value=0.01,
        max_value=1000000.0,
        value=150.0,
        step=0.01,
    )
    card4 = st.selectbox(
        "Card Network",
        ["visa", "mastercard", "american express", "discover"],
    )
    card6 = st.selectbox(
        "Card Type",
        ["debit", "credit"],
    )
    product = st.selectbox(
        "Product Code",
        ["W", "H", "C", "S", "R"],
        help="W=Web, H=Home, C=Card, S=Services, R=Retail"
    )

with col2:
    email = st.text_input(
        "Purchaser Email Domain",
        value="gmail.com",
        placeholder="gmail.com"
    )
    device_type = st.selectbox(
        "Device Type",
        ["desktop", "mobile"],
    )
    hour = st.slider(
        "Hour of Transaction (0-23)",
        min_value=0,
        max_value=23,
        value=14,
    )
    addr2 = st.number_input(
        "Billing Country Code",
        min_value=0,
        max_value=500,
        value=87,
    )

st.markdown("---")

# ------------------------------------------------------------------
# Advanced options
# ------------------------------------------------------------------
with st.expander("Advanced — Identity & Match Flags"):
    col3, col4 = st.columns(2)
    with col3:
        id_01 = st.number_input("Identity Score 1", value=0.0, step=0.1)
        id_02 = st.number_input("Identity Score 2", value=0.0, step=0.1)
        m4    = st.selectbox("Match Flag M4", ["T", "F", "unknown"])
    with col4:
        id_03 = st.number_input("Identity Score 3", value=0.0, step=0.1)
        id_06 = st.number_input("Identity Score 6", value=0.0, step=0.1)
        m6    = st.selectbox("Match Flag M6", ["T", "F", "unknown"])

# ------------------------------------------------------------------
# Submit
# ------------------------------------------------------------------
st.markdown("")
submit = st.button("🔍 Analyze Transaction", type="primary", use_container_width=True)

if submit:
    # Build payload
    payload = {
        "TransactionAmt": amount,
        "TransactionDT":  hour * 3600,
        "ProductCD":      product,
        "card4":          card4,
        "card6":          card6,
        "P_emaildomain":  email,
        "DeviceType":     device_type,
        "addr2":          addr2,
        "id_01":          id_01 if id_01 != 0.0 else None,
        "id_02":          id_02 if id_02 != 0.0 else None,
        "id_03":          id_03 if id_03 != 0.0 else None,
        "id_06":          id_06 if id_06 != 0.0 else None,
        "M4":             m4 if m4 != "unknown" else None,
        "M6":             m6 if m6 != "unknown" else None,
    }

    with st.spinner("Analyzing transaction..."):
        try:
            response = requests.post(
                API_URL,
                json=payload,
                timeout=30,
            )

            if response.status_code == 200:
                data = response.json()

                # Parse nested body if Lambda wraps it
                if 'body' in data:
                    result = json.loads(data['body'])
                else:
                    result = data

                score      = result.get('fraud_score', 0)
                is_fraud   = result.get('is_fraud', False)
                action     = result.get('action', 'allow')
                confidence = result.get('confidence', 'low')
                threshold  = result.get('threshold', 0.7)

                st.markdown("---")
                st.subheader("Result")

                # Action display
                if action == 'block':
                    st.error(f"🔴 **BLOCK** — High fraud risk detected")
                elif action == 'review':
                    st.warning(f"🟡 **REVIEW** — Suspicious transaction, manual review recommended")
                else:
                    st.success(f"🟢 **ALLOW** — Transaction appears legitimate")

                # Metrics
                col5, col6, col7 = st.columns(3)
                with col5:
                    st.metric("Fraud Score", f"{score:.4f}")
                with col6:
                    st.metric("Confidence", confidence.capitalize())
                with col7:
                    st.metric("Threshold", f"{threshold:.4f}")

                # Score bar
                st.markdown("**Risk Level:**")
                st.progress(min(score * 3, 1.0))

                # Details
                with st.expander("Full Response"):
                    st.json(result)

                with st.expander("Request Payload"):
                    st.json({k: v for k, v in payload.items() if v is not None})

            else:
                st.error(f"API Error: {response.status_code} — {response.text}")

        except requests.exceptions.Timeout:
            st.error("Request timed out — SageMaker endpoint may be warming up. Try again.")
        except Exception as e:
            st.error(f"Error: {str(e)}")

# ------------------------------------------------------------------
# Footer
# ------------------------------------------------------------------
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray; font-size: 0.8em'>
    IEEE-CIS Fraud Detection Dataset · XGBoost · AWS SageMaker · 
    Test AP: 0.7162 · Fraud Caught: 73.44%
    </div>
    """,
    unsafe_allow_html=True,
)