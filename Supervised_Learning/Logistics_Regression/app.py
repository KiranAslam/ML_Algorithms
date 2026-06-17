import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(
    page_title="Credit Risk Intelligence System",
    page_icon="🏦",
    layout="wide"
)

st.title("🏦 Credit Risk Intelligence System")

st.divider()
import os

@st.cache_resource
def load_credit_model():
    try:
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(BASE_DIR, "Data", "credit_scoring_model.pkl")
        
        return joblib.load(model_path)
    except FileNotFoundError:
        st.error("❌ `credit_scoring_model.pkl` not found! System looks at absolute path.")
        return None

model = load_credit_model()

age_woe_map = {0: 0.35, 1: 0.15, 2: -0.05, 3: -0.25, 4: -0.45}
util_woe_map = {0: -1.05, 1: -0.55, 2: 0.15, 3: 0.85, 4: 1.45}
income_woe_map = {0: 0.25, 1: 0.08, 2: -0.03, 3: -0.15, 4: -0.32}
debt_woe_map = {0: -0.12, 1: -0.02, 2: 0.05, 3: 0.18, 4: 0.38}

st.sidebar.header("⚙️ Risk Management Policy")
st.sidebar.markdown("Adjust the risk profile cutoff to simulation-match the bank's lending style.")

custom_threshold = st.sidebar.slider(
    "Decision Cutoff Threshold", 
    min_value=0.20, 
    max_value=0.80, 
    value=0.30, 
    step=0.05
)

st.sidebar.divider()
st.sidebar.write(f"**Current Rule Matrix:**")
st.sidebar.caption(f"• If Risk Probability > {custom_threshold*100:.0f}% ➔ **REJECT**")
st.sidebar.caption(f"• If Risk Probability ≤ {custom_threshold*100:.0f}% ➔ **APPROVE**")
st.subheader("👤 Applicant Demographics & Financial Parameters")
with st.form("credit_evaluation_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 Continuous Demographics & Ratios")
        age = st.number_input("Age of Applicant (Years)", min_value=18, max_value=100, value=35, step=1)
        monthly_income = st.number_input("Monthly Income ($)", min_value=0, value=5400, step=100)
        debt_ratio = st.number_input("Debt Ratio (Monthly Debt / Gross Income)", min_value=0.0, value=0.35, step=0.01)
        utilization = st.number_input("Revolving Credit Line Utilization (0.0 - 2.0+)", min_value=0.0, value=0.25, step=0.05)
   
    with col2:
        st.markdown("### 🚨 Historical Delinquency Counts")
        past_due_30_59 = st.number_input("Times 30-59 Days Past Due (No Worse)", min_value=0, max_value=20, value=0, step=1)
        past_due_60_89 = st.number_input("Times 60-89 Days Past Due (No Worse)", min_value=0, max_value=20, value=0, step=1)
        past_due_90_late = st.number_input("Times 90+ Days Late / Serious Default", min_value=0, max_value=20, value=0, step=1)

    submit_application = st.form_submit_button("Run Strategic Risk Assessment")



    if model is None:
        st.error("Execution halted. Model state cannot be initialized.")
    else:
        st.subheader("📊 Analytical Risk Assessment Report")
        
        with st.spinner("Processing pipeline matrices..."):
            if age <= 35: age_bin = 0
            elif age <= 45: age_bin = 1
            elif age <= 55: age_bin = 2
            elif age <= 65: age_bin = 3
            else: age_bin = 4
            age_WoE = age_woe_map.get(age_bin, 0.0)
            clean_utilization = 1.0 if utilization > 1.0 else utilization
            if clean_utilization <= 0.03: util_bin = 0
            elif clean_utilization <= 0.15: util_bin = 1
            elif clean_utilization <= 0.45: util_bin = 2
            elif clean_utilization <= 0.80: util_bin = 3
            else: util_bin = 4
            util_WoE = util_woe_map.get(util_bin, 0.0)
            if monthly_income <= 3400: inc_bin = 0
            elif monthly_income <= 5400: inc_bin = 1
            elif monthly_income <= 7400: inc_bin = 2
            elif monthly_income <= 11000: inc_bin = 3
            else: inc_bin = 4
            income_WoE = income_woe_map.get(inc_bin, 0.0)
            if debt_ratio <= 0.18: debt_bin = 0
            elif debt_ratio <= 0.36: debt_bin = 1
            elif debt_ratio <= 0.55: debt_bin = 2
            elif debt_ratio <= 1.35: debt_bin = 3
            else: debt_bin = 4
            debt_WoE = debt_woe_map.get(debt_bin, 0.0)
            input_vector = pd.DataFrame([{
                'age_WoE': age_WoE,
                'MonthlyIncome_WoE': income_WoE,
                'DebtRatio_WoE': debt_WoE,
                'RevolvingUtilizationOfUnsecuredLines_WoE': util_WoE,
                'NumberOfTime30-59DaysPastDueNotWorse': past_due_30_59,
                'NumberOfTimes90DaysLate': past_due_90_late,
                'NumberOfTime60-89DaysPastDueNotWorse': past_due_60_89
            }])
            calculated_probability = model.predict_proba(input_vector)[0][1]

        m_col1, m_col2 = st.columns(2)
        with m_col1:
            st.metric(
                label="Calculated Default Probability Score", 
                value=f"{calculated_probability * 100:.2f}%",
                delta=f"Policy Cutoff: {custom_threshold * 100:.0f}%",
                delta_color="inverse"
            )
            
        with m_col2:
            if calculated_probability > custom_threshold:
                st.error("🚨 STATUS: APPLICATION REJECTED (High Credit Defaulter Risk)")
            else:
                st.success("✅ STATUS: APPLICATION APPROVED (Safe Risk Criteria Profile)")
        with st.expander("🔎 View Internal Feature Processing State (WoE Vectors)"):
            st.json({
                "Mapped Age WoE": age_WoE,
                "Mapped Income WoE": income_WoE,
                "Mapped Debt Ratio WoE": debt_WoE,
                "Capped & Mapped Utilization WoE": util_WoE,
                "Raw Delinquency Features Loaded": {
                    "30-59 Days Past Due": past_due_30_59,
                    "60-89 Days Past Due": past_due_60_89,
                    "90+ Days Late": past_due_90_late
                }
            })