import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load('loan_risk_model.pkl')

# App title
st.title("Loan Default Risk Prediction App")

# Sidebar inputs
st.sidebar.header("Enter Borrower Details")

age = st.sidebar.number_input("Age", min_value=18, max_value=100, value=30)
income = st.sidebar.number_input("Monthly Income", value=50000)
loan_amount = st.sidebar.number_input("Loan Amount", value=300000)
loan_tenure = st.sidebar.number_input("Loan Tenure (months)", value=36)
interest_rate = st.sidebar.number_input("Interest Rate (%)", value=12.0)
collateral_value = st.sidebar.number_input("Collateral Value", value=200000)
outstanding_loan = st.sidebar.number_input("Outstanding Loan Amount", value=100000)
monthly_emi = st.sidebar.number_input("Monthly EMI", value=15000)
missed_payments = st.sidebar.number_input("Missed Payments", value=0)
days_past_due = st.sidebar.number_input("Days Past Due", value=0)

# Input DataFrame
input_df = pd.DataFrame({
    'Age': [age],
    'Monthly_Income': [income],
    'Loan_Amount': [loan_amount],
    'Loan_Tenure': [loan_tenure],
    'Interest_Rate': [interest_rate],
    'Collateral_Value': [collateral_value],
    'Outstanding_Loan_Amount': [outstanding_loan],
    'Monthly_EMI': [monthly_emi],
    'Num_Missed_Payments': [missed_payments],
    'Days_Past_Due': [days_past_due]
})

# Predict Button
if st.button("Predict Risk"):
    risk_score = model.predict_proba(input_df)[0][1]
    risk_flag = int(risk_score > 0.4)

    st.subheader("Prediction Result")
    st.write(f"Risk Score: **{risk_score:.2f}**")

    if risk_flag == 1:
        st.error("High Risk of Default")
    else:
        st.success("Low Risk of Default")

    # Recovery strategy
    if risk_score > 0.75:
        strategy = "Immediate legal notices & aggressive recovery attempts"
    elif 0.40 <= risk_score <= 0.75:
        strategy = "Settlement offers & repayment plans"
    else:
        strategy = "Automated reminders & monitoring"

    st.info(f"Recommended Recovery Strategy: **{strategy}**")
