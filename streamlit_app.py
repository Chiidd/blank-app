
import streamlit as st
import pandas as pd
import joblib

# Load model and preprocessor
pipeline = joblib.load("pipeline.pkl")


st.title("🛒 Supermarket Customer Rating Predictor")

st.write("Enter transaction details to predict customer rating")

# User inputs
unit_price = st.number_input("Unit Price", min_value=0.0)
quantity = st.number_input("Quantity", min_value=1)
total = st.number_input("Total Purchase", min_value=0.0)
cogs = st.number_input("Cost of Goods Sold", min_value=0.0)
gross_income = st.number_input("Gross Income", min_value=0.0)

# Convert input into dataframe
input_data = pd.DataFrame({
    "Unit price": [unit_cost],
    "Quantity": [quantity],
    "Total": [total],
    "cogs": [cogs],
    "gross income": [revenue]
})

# Predict
prediction = pipeline.predict(input_data)

# Predict
# if st.button("Predict Rating"):
 # prediction = pipeline.predict(input_scaled)
# st.success(f"Predicted Customer Rating: {prediction[0]:.2f}")
 
