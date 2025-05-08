import streamlit as st
import pandas as pd
import pickle
import xgboost
import shap
import matplotlib.pyplot as plt

with open('HR.sav', 'rb') as file:
    data = pickle.load(file)

st.title("HR Employee Attrition")
st.image("https://miro.medium.com/v2/resize:fit:1100/format:webp/1*hVmDd7kBxo2z2FmH8Auvlg.png", width=500)
st.header("Fill in the Employee Information")

# [INPUT SECTIONS REMAIN THE SAME AS YOUR CODE]

# --- Prediction ---
selected_items = pd.DataFrame({
    "age": [age],
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
    pre = data.predict(selected_items)
    result = "Yes" if pre[0] == 1 else "No"
    st.success(f"Predicted attrition: **{result}**")

    if result == "Yes":
        st.warning("The employee is at risk of leaving. Suggested improvements:")

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

    # --- SHAP Explanation ---
    st.subheader("Why this prediction?")
    explainer = shap.Explainer(data)
    shap_values = explainer(selected_items)

    fig, ax = plt.subplots()
    shap.plots.waterfall(shap_values[0], max_display=10, show=False)
    st.pyplot(fig)
