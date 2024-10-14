# 🎗️ 𝐏𝐫𝐨𝐣𝐞𝐜𝐭 𝐒𝐩𝐨𝐭𝐥𝐢𝐠𝐡𝐭: 𝐓𝐞𝐥𝐞𝐜𝐨𝐦 𝐂𝐮𝐬𝐭𝐨𝐦𝐞𝐫 𝐂𝐡𝐮𝐫𝐧 𝐏𝐫𝐞𝐝𝐢𝐜𝐭𝐢𝐨𝐧 𝐰𝐢𝐭𝐡 𝐌𝐚𝐜𝐡𝐢𝐧𝐞 𝐋𝐞𝐚𝐫𝐧𝐢𝐧𝐠 🔍

### 📋 𝐏𝐫𝐨𝐣𝐞𝐜𝐭 𝐎𝐯𝐞𝐫𝐯𝐢𝐞𝐰

I recently completed a Telecom Customer Churn Prediction project using various machine learning algorithms. The goal was to predict whether a customer will churn based on their demographic, service usage, and billing data. This involved data preprocessing, training models, evaluating their performance, and saving the models for future use. I experimented with different algorithms to maximize prediction accuracy and enhance business value.

### 🛠️ 𝐓𝐨𝐨𝐥𝐬 𝐔𝐬𝐞𝐝

**Programming Language:** Python <br>
**Libraries:** <br>
* NumPy: For numerical computations.<br>
* Pandas: For data manipulation and analysis.<br>
* Scikit-learn: For building and evaluating models.<br>
* Joblib: For saving and loading models.<br>
**Streamlit:** For building an interactive web app.<br>

### 🔍 𝐊𝐞𝐲 𝐒𝐭𝐞𝐩𝐬

* Data Preparation: Loaded and cleaned the dataset, handled missing values, and performed feature encoding for categorical variables like InternetService, Contract, and PaymentMethod. Also scaled numerical features like MonthlyCharges and TotalCharges.<br>

* Model Training: Trained multiple machine learning models—Gradient Boosting, Logistic Regression, Decision Tree, Random Forest, and Support Vector Classifier (SVC)—to predict customer churn.<br>

* Prediction: Used these models to predict whether customers are likely to churn based on their input data.<br>

* Performance Evaluation: Evaluated model performance using metrics such as accuracy scores, confusion matrices, and classification reports.<br>

* Model Deployment: Built an interactive app using Streamlit to allow users to input customer data and receive real-time churn predictions.<br>

### 📊 𝐊𝐞𝐲 𝐅𝐢𝐧𝐝𝐢𝐧𝐠𝐬

**Gradient Boosting Classifier:** <br>
* Accuracy: 82.36%<br>
* Precision: 0.79, Recall: 0.74, F1-Score: 0.76<br>
**Logistic Regression:** <br>
* Accuracy: 80.50%<br>
* Precision: 0.78, Recall: 0.72, F1-Score: 0.75<br>
**Random Forest:** <br>
* Accuracy: 81.80%<br>
* Precision: 0.78, Recall: 0.73, F1-Score: 0.75<br>
**Support Vector Classifier (SVC):** <br>
* Accuracy: 80.12%<br>
* Precision: 0.77, Recall: 0.72, F1-Score: 0.74<br>

### 🏁 𝐂𝐨𝐧𝐜𝐥𝐮𝐬𝐢𝐨𝐧

The Gradient Boosting Classifier provided the best performance for predicting customer churn. Each model gave valuable insights into how customer behavior correlates with churn. Using these predictions, telecom companies can tailor retention strategies more effectively.

### 🌐 𝐀𝐩𝐩𝐥𝐢𝐜𝐚𝐭𝐢𝐨𝐧

This app could be used by telecom companies to proactively identify customers likely to churn, enabling targeted offers or interventions to retain them. The model's predictions could significantly enhance customer retention efforts and improve overall business outcomes.

https://customerchurnprediction-j2hqdsygi3hm6edgkel8by.streamlit.app/
