import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

st.set_page_config(page_title="AI Student Performance Prediction")

st.title("AI-Based Student Performance Prediction")
st.write("Predict a student's final score using simple academic and study-related inputs.")

@st.cache_data
def load_data():
    return pd.read_csv("student_data.csv")

@st.cache_resource
def train_model():
    data = load_data()
    X = data.drop("final_score", axis=1)
    y = data["final_score"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    mae = mean_absolute_error(y_test, prediction)
    return model, mae

model, mae = train_model()

st.sidebar.header("Student Details")
study_hours = st.sidebar.number_input("Study Hours per Day", 0.0, 12.0, 4.0)
attendance = st.sidebar.number_input("Attendance (%)", 0.0, 100.0, 80.0)
previous_score = st.sidebar.number_input("Previous Score", 0.0, 100.0, 70.0)
assignments = st.sidebar.number_input("Assignments Completed", 0, 20, 10)

input_data = pd.DataFrame({
    "study_hours": [study_hours],
    "attendance": [attendance],
    "previous_score": [previous_score],
    "assignments_completed": [assignments]
})

if st.button("Predict Final Score"):
    result = model.predict(input_data)[0]
    result = max(0, min(100, result))
    st.success(f"Predicted Final Score: {result:.2f}%")

st.info(f"Model Mean Absolute Error on test data: {mae:.2f}")
