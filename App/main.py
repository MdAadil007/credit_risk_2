import streamlit as st
from prediction_helper import predict

st.set_page_config(page_title="Credit Risk Modelling", page_icon="🏦", layout="wide")

st.markdown("## 🏦 Credit Risk Modelling")
st.caption("Fill in the applicant details to predict default probability and credit score.")
st.divider()

row1 = st.columns(3)
row2 = st.columns(3)
row3 = st.columns(3)
row4 = st.columns(3)

with row1[0]:
    age = st.number_input('Age', min_value=18, step=1, max_value=100, value=28)
with row1[1]:
    income = st.number_input('Income (₹)', min_value=0, value=1200000)
with row1[2]:
    loan_amount = st.number_input('Loan Amount (₹)', min_value=0, value=2560000)

loan_to_income_ratio = loan_amount / income if income > 0 else 0
with row2[0]:
    st.markdown("**Loan to Income Ratio**")
    st.markdown(f"## {loan_to_income_ratio:.2f}")

with row2[1]:
    loan_tenure_months = st.number_input('Loan Tenure (months)', min_value=0, step=1, value=36)
with row2[2]:
    avg_dpd_per_delinquency = st.number_input('Avg DPD per Delinquency', min_value=0, max_value=10, value=2)

with row3[0]:
    # trained on a 0-1 fraction (delinquent_months / total_loan_months), not a 0-100 percentage
    delinquency_ratio = st.slider(
        'Delinquency Ratio',
        min_value=0.0, max_value=1.0, value=0.30, step=0.01,
        help="Fraction of loan tenure months in which the applicant was delinquent (0 = never, 1 = always)."
    )
with row3[1]:
    credit_utilization_ratio = st.number_input('Credit Utilization Ratio (%)', min_value=0, max_value=99, step=1, value=30)
with row3[2]:
    num_open_accounts = st.number_input('Open Loan Accounts', min_value=1, max_value=4, step=1, value=2)

with row4[0]:
    # Mortgage is the model's dropped baseline category, kept here for realistic UI
    residence_type = st.selectbox('Residence Type', ['Owned', 'Rented', 'Mortgage'])
with row4[1]:
    # Auto is the model's dropped baseline category, kept here for realistic UI
    loan_purpose = st.selectbox('Loan Purpose', ['Education', 'Home', 'Auto', 'Personal'])
with row4[2]:
    loan_type = st.selectbox('Loan Type', ['Unsecured', 'Secured'])

st.divider()

calculate = st.button('🔍 Calculate Risk', use_container_width=True)

if calculate:
    probability, credit_score, rating = predict(
        age, income, loan_amount, loan_tenure_months, avg_dpd_per_delinquency,
        delinquency_ratio, credit_utilization_ratio, num_open_accounts,
        residence_type, loan_purpose, loan_type
    )

    st.markdown("### 📊 Results")

    res_col1, res_col2, res_col3 = st.columns(3)
    with res_col1:
        st.markdown("**Default Probability**")
        st.markdown(f"## {probability:.2%}")
    with res_col2:
        st.markdown("**Credit Score**")
        st.markdown(f"## {credit_score} / 900")
    with res_col3:
        st.markdown("**Rating**")
        st.markdown(f"## {rating}")

    if rating == 'Excellent':
        st.success(f"✅ Excellent ({credit_score}) — Low risk applicant.")
    elif rating == 'Good':
        st.info(f"ℹ️ Good ({credit_score}) — Acceptable risk.")
    elif rating == 'Average':
        st.warning(f"⚠️ Average ({credit_score}) — Elevated risk. Review carefully.")
    else:
        st.error(f"🚫 Poor ({credit_score}) — High risk applicant.")