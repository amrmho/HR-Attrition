import streamlit as st
import pandas as pd
import pickle
import xgboost
import shap
import matplotlib.pyplot as plt

# Load the trained model
with open('HR.sav', 'rb') as file:
    data = pickle.load(file)

# Streamlit UI for inputs
st.title("HR Employee Attrition")
st.image("https://miro.medium.com/v2/resize:fit:1100/format:webp/1*hVmDd7kBxo2z2FmH8Auvlg.png", width=500)
st.header("Fill in the Employee Information")

# --- Personal Info ---
with st.expander("🧍 Personal Information"):
    c1, c2, c3 = st.columns(3)
    age = c1.number_input("Age", 18, 60)
    gender_selected = c2.selectbox("Gender", ['Female', 'Male'])
    gender_selected_id = {'Female': 0, 'Male': 1}[gender_selected]
    MaritalStatus_selected = c3.selectbox("Marital Status", ['Single', 'Married', 'Divorced'])
    MaritalStatus_selected_id = {'Single': 2, 'Married': 1, 'Divorced': 0}[MaritalStatus_selected]

# --- Other input sections go here ---

# Creating the selected_items DataFrame for prediction
selected_items = pd.DataFrame({
    "Age": [age],
    "BusinessTravel": [BusinessTravel_selected_id],
    "Department": [Department_selected_id],
    "Education": [Education],
    "EducationField": [EducationField_selected_id],
    "EnvironmentSatisfaction": [EnvironmentSatisfaction],
    "Gender": [gender_selected_id],
    "HourlyRate": [HourlyRate],
    "JobInvolvement": [JobInvolvement],
    "JobLevel": [JobLevel],
    "JobRole": [JobRole_selected_id],
    "JobSatisfaction": [JobSatisfaction],
    "MaritalStatus": [MaritalStatus_selected_id],
    "OverTime": [OverTime_selected_id],
    "PerformanceRating": [PerformanceRating],
    "RelationshipSatisfaction": [RelationshipSatisfaction],
    "StockOptionLevel": [StockOptionLevel],
    "TrainingTimesLastYear": [TrainingTimesLastYear],
    "WorkLifeBalance": [WorkLifeBalance],
    "DailyRateCategory": [DailyRate_selected_id],
    "DistanceFromHomeCategory": [DistanceFromHome_selected_id],
    "MonthlyIncomeCategory": [MonthlyIncomeCategory_selected_id],
    "NumCompaniesWorkedCategory": [NumCompaniesWorkedCategory_selected_id],
    "TotalWorkingYearsCategory": [TotalWorkingYearsCategory_selected_id],
    "YearsAtCompanyCategory": [YearsAtCompanyCategory_selected_id],
    "YearsInCurrentRoleCategory": [YearsInCurrentRoleCategory_selected_id],
    "YearsSinceLastPromotionCategory": [YearsSinceLastPromotionCategory_selected_id],
    "YearsWithCurrManagerCategory": [YearsWithCurrManagerCategory_selected_id]
}, index=[0])

if st.button("Predict attrition"):
    # Make the prediction
    pre = data.predict(selected_items)
    result = "Yes" if pre[0] == 1 else "No"
    st.success(f"🔍 Predicted attrition: **{result}**")

    # SHAP Explanation
    explainer = shap.Explainer(data)  # Create the SHAP explainer
    shap_values = explainer(selected_items)  # Get the SHAP values for the selected input data

    # Plot the SHAP values
    st.subheader("Why this prediction?")
    fig, ax = plt.subplots()
    shap.plots.waterfall(shap_values[0], max_display=10, show=False)
    st.pyplot(fig)

    # Suggestions for improvement (if attrition is predicted)
    if result == "Yes":
        suggestions = []

        if JobSatisfaction < 3:
            suggestions.append("Increase job satisfaction.")
        if WorkLifeBalance < 3:
            suggestions.append("Improve work-life balance.")
        if OverTime_selected_id == 1:
            suggestions.append("Reduce overtime hours.")
        if YearsSinceLastPromotionCategory_selected_id >= 3:
            suggestions.append("Consider promotion opportunities.")
        if EnvironmentSatisfaction < 3:
            suggestions.append("Enhance work environment.")
        if RelationshipSatisfaction < 3:
            suggestions.append("Improve interpersonal relationships.")

        st.markdown("### Recommended Actions:")
        for tip in suggestions:
            st.markdown(f"- {tip}")
