# pip install streamlit: pip install streamlit
import streamlit as st
from src.pipelines.prediction_pipeline import CustomData, PredictPipeline

# Custom CSS for styling
st.markdown(
    """
    <style>
        /* Custom CSS */
        .custom-title {
            color: #0b72b9;
            font-size: 40px;
            text-align: center;
        }
        .custom-button {
            background-color: #0b72b9;
            color: white;
        }
    </style>
    """, 
    unsafe_allow_html=True
)

# Removed the selectbox for UI Type and always use Professional UI
#st.markdown("<h1 class='custom-title'>Customer Churn Prediction</h1>", unsafe_allow_html=True)

# New: Centered logo display with reduced size and updated parameter
col1, col2, col3 = st.columns(3)
with col2:
    st.image("logo/logo.png", width=100)  # replaced deprecated use_column_width with width
# Removed the selectbox for UI Type and always use Professional UI
st.markdown("<h1 class='custom-title'>Customer Churn Prediction</h1>", unsafe_allow_html=True)


# New: Add a section listing application features
st.sidebar.markdown("### Application Features")
st.sidebar.markdown(
    """
    - Interactive customer data input form
    - Real-time prediction using an ANN model
    - Customized styling for a professional look
    - Clear success and error messaging based on prediction
    """
)

with st.form("churn_form", clear_on_submit=True):
    credit_score = st.number_input("Credit Score", value=600, step=1)
    geography = st.selectbox("Geography", options=["Spain", "France", "Germany"])
    gender = st.selectbox("Gender", options=["Male", "Female"], index=0)
    age = st.number_input("Age", value=40, step=1)
    tenure = st.number_input("Tenure", value=5, step=1)
    balance = st.number_input("Balance", value=10000.0, step=0.01)
    num_of_products = st.number_input("Number of Products", value=1, step=1)
    has_cr_card = st.number_input("Has Credit Card (1/0)", value=1, step=1)
    is_active_member = st.number_input("Is Active Member (1/0)", value=1, step=1)
    estimated_salary = st.number_input("Estimated Salary", value=50000.0, step=0.01)
    submitted = st.form_submit_button("Predict")

if submitted:
    data = CustomData(
        credit_score=credit_score,
        geography=geography,
        gender=gender,
        age=age,
        tenure=tenure,
        balance=balance,
        num_of_products=num_of_products,
        has_cr_card=has_cr_card,
        is_active_member=is_active_member,
        estimated_salary=estimated_salary
    )
    pipeline = PredictPipeline()
    result = pipeline.predict(data)
    if result[0] == 1:
        st.error("Alert: The customer is likely to churn.")
    else:
        st.success("Success: The customer is not likely to churn.")