import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import base64
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import os

# Function to read and encode the image file
def get_base64_image(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    return encoded_string

# Set the background image using CSS
def set_background(image_base64):
    page_bg_img = f"""
    <style>
    .stApp {{
        background: url("data:image/png;base64,{image_base64}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        color: white;
    }}
    .css-1g8v9l0 {{
        background: rgba(255, 255, 255, 0.8);
        padding: 20px;
        border-radius: 10px;
    }}
    .stButton > button {{
        background-color: #4C4C6D;
        color: white;
    }}
    .stButton > button:hover {{
        background-color: #6A5ACD;
        color: white;
    }}
    .stSlider > div {{
        background-color: transparent;
    }}
    .stSelectbox div {{
        color: white;
    }}
    .stSubheader {{
        color: white;
    }}
    </style>
    """
    st.markdown(page_bg_img, unsafe_allow_html=True)

# Call the function with the uploaded background image
image_base64 = get_base64_image("image.jpg")  # Path to your uploaded image
set_background(image_base64)


# Load the trained model
model = joblib.load("final_gb_classifier.pkl")

data = pd.read_csv('Telco-Customer-Churn.csv')

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

st.subheader("📊 Dataset Preview:")
st.dataframe(data.head())



st.info("""
    **Note:** The model has been trained using this dataset, and the predictions made by the app are based on the provided customer information.
""")


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
# Count the number of customers who churned vs. not churned
churn_counts = data['Churn'].value_counts()  # Assuming 'Churn' column is binary (0 for Not Churn, 1 for Churn)

# Bar Chart to visualize churn vs not churn
def plot_churn_vs_not_churn_dataset(churn_counts):
    plt.figure(figsize=(8, 6))
    sns.barplot(x=churn_counts.index, y=churn_counts.values, palette="pastel")
    plt.title('Total Customers: Churn vs Not Churn')
    plt.xlabel('Customer Status')
    plt.ylabel('Number of Customers')
    plt.xticks([0, 1], ['Not Churn', 'Churn'])  # Assuming 0 is Not Churn and 1 is Churn
    st.pyplot(plt)

# Pie Chart to visualize churn vs not churn
def plot_churn_pie_chart(churn_counts):
    fig, ax = plt.subplots()
    ax.pie(churn_counts, labels=['Not Churn', 'Churn'], autopct='%1.1f%%', startangle=90, colors=['#FFB6C1', '#99C5C4'])
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    st.pyplot(fig)





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
    st.success(f"The Probability of the customer churning is: {churn_probability:.2f}")
    
    
    
col1,col2 = st.columns(2)

with col1:
    # Streamlit UI (rest of your app remains the same)
    st.subheader("Churn vs Not Churn BarChart") 
    # Display the bar chart
    plot_churn_vs_not_churn_dataset(churn_counts)
    
with col2:
    # Optionally, display the pie chart
    st.subheader("Churn vs Not Churn PieChart")
    plot_churn_pie_chart(churn_counts)
        

