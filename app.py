import streamlit as st
import pandas as pd
import joblib

# Page configuration
st.set_page_config(
    page_title="Electric Bill Predictor",
    page_icon="⚡",
    layout="centered"
)

# Load the saved model pipeline (.pkl)
@st.cache_resource
def load_model():
    return joblib.load('electric_bill_model.pkl')

model = load_model()

# App UI Header
st.title("⚡ Electric Bill Predictor")
st.write("Enter the number of **AC Units** consumed to predict your estimated electricity bill using a trained Polynomial Regression model.")

# User input form
with st.form("prediction_form"):
    ac_units = st.number_input(
        "AC Units Consumed", 
        min_value=0.0, 
        max_value=500.0, 
        value=30.0, 
        step=1.0,
        help="Enter the total units consumed by your AC."
    )
    
    submit_button = st.form_submit_button(label="Predict Bill")

# Prediction logic
if submit_button:
    # Format input into a DataFrame matching model expectation
    input_data = pd.DataFrame([[ac_units]], columns=['AC_Units'])
    
    # Predict using the loaded model
    predicted_bill = model.predict(input_data)[0]
    
    # Display result
    st.success(f"### Estimated Electric Bill: ₹ {predicted_bill:,.2f}")
    st.info(f"Based on **{ac_units}** AC units consumed.")

# Footer info
st.markdown("---")
st.caption("Developed with Streamlit & Scikit-Learn | Christ College Project")
