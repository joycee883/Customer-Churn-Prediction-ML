import streamlit as st
import pandas as pd
import numpy as np
import joblib
import base64
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


# Function to set background image
def set_background(image_path):
    with open(image_path, "rb") as image_file:
        data = image_file.read()
    encoded_image = base64.b64encode(data).decode()  # Encode the image to base64
    background_style = f"""
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{encoded_image}");
        background-size: cover;
    }}
    </style>
    """
    st.markdown(background_style, unsafe_allow_html=True)

# Example usage
set_background(r"C:\Users\91939\Desktop\AI&DS\Data science projects\CustomerChurnPrediction\background-images_www-1-1.png")


# Load the trained model
model = joblib.load("final_gb_classifier.pkl")

# Function to preprocess input data
def preprocess_input(data):
    # Convert input data to DataFrame
    df = pd.DataFrame(data, index=[0])
    
    # Convert categorical variables to numeric
    df['InternetService'] = df['InternetService'].map({'DSL': 0, 'Fiber optic': 1, 'No': 2})
    df['Contract'] = df['Contract'].map({'Month-to-month': 0, 'One year': 1, 'Two year': 2})
    df['PaymentMethod'] = df['PaymentMethod'].map({'Electronic check': 0, 'Mailed check': 1, 'Bank transfer (automatic)': 2, 'Credit card (automatic)': 3})
    
    # Return preprocessed DataFrame
    return df

# Streamlit UI
st.title("📱 Telecom Customer Retention Prediction ")

st.markdown("""
    <div style="text-align: center;">
    </div>
    """, unsafe_allow_html=True)
    
st.subheader("📝 Description:")
st.write("📡 The **Telecom Customer Churn Prediction** app helps telecom companies predict which customers are likely to leave (churn) based on demographic, service, and billing information."
"Using a pre-trained machine learning model, the app allows users to input customer details such as service type, contract, and charges, and provides a prediction of whether the customer will stay or churn. This tool aids in targeted retention strategies to reduce customer turnover.")

# Collect user inputs in the sidebar
st.sidebar.title("🔍 Customer Input Features")

gender = st.sidebar.selectbox("Gender", [0, 1], help="0 = Male, 1 = Female 👫")
senior_citizen = st.sidebar.selectbox("Senior Citizen", [0, 1], help="0 = No, 1 = Yes 👴👵")
partner = st.sidebar.selectbox("Partner", [0, 1], help="0 = No, 1 = Yes 💑")
dependents = st.sidebar.selectbox("Dependents", [0, 1], help="0 = No, 1 = Yes 👨‍👩‍👧")
phone_service = st.sidebar.selectbox("Phone Service", [0, 1], help="0 = No, 1 = Yes 📞")
multiple_lines = st.sidebar.selectbox("Multiple Lines", [0, 1], help="0 = No, 1 = Yes 🧑‍🤝‍🧑")
internet_service = st.sidebar.selectbox("Internet Service", ['DSL', 'Fiber optic', 'No'], help="Customer's internet service provider 🌐")
online_security = st.sidebar.selectbox("Online Security", [0, 1, 2], help="0 = No, 1 = Yes, 2 = No internet service 🔒")
online_backup = st.sidebar.selectbox("Online Backup", [0, 1, 2], help="0 = No, 1 = Yes, 2 = No internet service 💾")
device_protection = st.sidebar.selectbox("Device Protection", [0, 1, 2], help="0 = No, 1 = Yes, 2 = No internet service 🛡️")
tech_support = st.sidebar.selectbox("Tech Support", [0, 1, 2], help="0 = No, 1 = Yes, 2 = No internet service 🖥️")
streaming_tv = st.sidebar.selectbox("Streaming TV", [0, 1], help="0 = No, 1 = Yes 📺")
streaming_movies = st.sidebar.selectbox("Streaming Movies", [0, 1], help="0 = No, 1 = Yes 🎬")
contract = st.sidebar.selectbox("Contract", ['Month-to-month', 'One year', 'Two year'], help="The contract term of the customer 📜")
paperless_billing = st.sidebar.selectbox("Paperless Billing", [0, 1], help="0 = No, 1 = Yes 🧾")
payment_method = st.sidebar.selectbox("Payment Method", ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'], help="The customer's payment method 💳")
monthly_charges = st.sidebar.number_input("Monthly Charges", value=0.0, help="The amount charged to the customer monthly 💰")
total_charges = st.sidebar.number_input("Total Charges", value=0.0, help="The total amount charged to the customer 💵")
tenure_group = st.sidebar.number_input("Tenure Group", value=0, help="Number of months the customer has stayed with the company 🕰️")





# Make prediction
if st.sidebar.button("🔮 Predict"):
    user_data = {
        'gender': gender,
        'SeniorCitizen': senior_citizen,
        'Partner': partner,
        'Dependents': dependents,
        'PhoneService': phone_service,
        'MultipleLines': multiple_lines,
        'InternetService': internet_service,
        'OnlineSecurity': online_security,
        'OnlineBackup': online_backup,
        'DeviceProtection': device_protection,
        'TechSupport': tech_support,
        'StreamingTV': streaming_tv,
        'StreamingMovies': streaming_movies,
        'Contract': contract,
        'PaperlessBilling': paperless_billing,
        'PaymentMethod': payment_method,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges,
        'tenure_group': tenure_group
    }
    
    # Preprocess input data
    processed_data = preprocess_input(user_data)
    
    # Make prediction
    prediction = model.predict(processed_data)
    
    churn_probability = model.predict_proba(processed_data)[0][1]
    # Display prediction result
    if prediction[0] == 1:
       st.error("🚨 The customer is likely to churn.")
    else:
       st.success("✅ The customer is likely to stay.")
     
    st.subheader("🎯 Churn Probability:")
    st.write(f"The Probability of the customer churning is: {churn_probability:.2f}")

        

