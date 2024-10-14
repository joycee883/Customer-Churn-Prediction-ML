# 1.import libraries for data manipulation and mathematical arrays
import numpy as np
import pandas as pd 

# 2.import libraries for data visualizations 
import matplotlib.pyplot as plt
import seaborn as sns

# 3.Load the dataset
telco = pd.read_csv(r'C:\Users\91939\Desktop\AI&DS\Data science projects\CustomerChurnPrediction\Telco-Customer-Churn.csv')

# 4.Perform basic operations
# print the top 5 records of the dataset
print("Top 5 records of the Dataset:" ,telco.head())

# print the shape of the dataset
print("Shape of the Dataset:" ,telco.shape)

# print the info of the dataset
print("Info of the Dataset:" ,telco.info())

# Check for null values
print("Checking for the null values in the Dataset:" ,telco.isna().sum())

# print the columns in the dataset
print("Columns in the dataset" ,telco.columns)

for col in telco.columns:
    print("column:{} - Unique values:{}".format(col,telco[col].unique()))

# 5.Convert categorical to numeric
telco.TotalCharges = pd.to_numeric(telco.TotalCharges, errors='coerce')

# print the dtypes of the attributes
print(telco.dtypes)

# print descriptive stats of numerical attributes
telco.describe()

# plot the value counts of churn Yes,No
telco['Churn'].value_counts().plot(kind='bar')
plt.xlabel("Count")
plt.ylabel("Churn")
plt.title("Count of Churn")  
plt.show()

# print the percentage of each class in churn
print(telco['Churn'].value_counts()/len(telco)*100)

# copy the data 
telco_data = telco.copy()

# check for null values
print(telco_data.isna().sum())

# print the records with null values
telco_data[telco_data['TotalCharges'].isna()==True]

# check for the percentage of null values
telco_data.isna().sum()/len(telco_data)

# 6.Missing value treatment
# since the percentage is only 0.001562 we can ignore them
telco_data.dropna(how = 'any', inplace = True)

# Get the max tenure
print(telco_data['tenure'].max())

# Define the bins and labels
bins = [0, 12, 24, 36, 48, 60, 72]
labels = ['1 - 12', '13 - 24', '25 - 36', '37 - 48', '49 - 60', '61 - 72']

# Create the tenure_group column
telco_data['tenure_group'] = pd.cut(telco_data['tenure'], bins=bins, labels=labels, right=False)

# print how many customers stayed in the above groups
telco_data['tenure_group'].value_counts()

# print the percentage for the above code
telco_data['tenure_group'].value_counts()/len(telco_data)

# 7.Remove columns not required for processing
#drop column customerID and tenure
telco_data.drop(columns= ['customerID','tenure'], axis=1, inplace=True)

telco_data.columns

# 8.individual count plots for each predictor, showing the distribution of 'Churn' across different features, excluding 'TotalCharges' and 'MonthlyCharges'.
for i, predictor in enumerate(telco_data.drop(columns=['Churn', 'TotalCharges', 'MonthlyCharges'])):
    plt.figure(i)
    sns.countplot(data=telco_data, x=predictor, hue='Churn')

# 9.Convert the target variable 'Churn' in a binary numeric variable i.e. Yes=1 ; No = 0
telco_data['Churn'] = np.where(telco_data.Churn == 'Yes',1,0)

# 10.Convert all the categorical variables into dummy variables

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()

categ=['gender','SeniorCitizen', 'tenure_group' ,'Partner', 'Dependents', 'PhoneService',
       'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
       'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
       'Contract', 'PaperlessBilling', 'PaymentMethod',  'Churn',]

telco_data[categ] = telco_data[categ].apply(le.fit_transform)
telco_data['PaymentMethod'].value_counts()

# Relationship between Monthly Charges and Total Charges
sns.boxplot(data=telco_data[['TotalCharges', 'MonthlyCharges']])

sns.lmplot(data=telco_data, x='MonthlyCharges', y='TotalCharges', fit_reg=False)

# 11.Churn by Monthly Charges and Total Charges
# kernel density estimate (KDE) plot.
Mth = sns.kdeplot(telco_data.MonthlyCharges[(telco_data["Churn"] == 0) ],
                color="Red", fill = True)
Mth = sns.kdeplot(telco_data.MonthlyCharges[(telco_data["Churn"] == 1) ],
                ax =Mth, color="Blue", shade= True)
Mth.legend(["No Churn","Churn"],loc='upper right')
Mth.set_ylabel('Density')
Mth.set_xlabel('Monthly Charges')
Mth.set_title('Monthly charges by churn')
print(Mth)

Tot = sns.kdeplot(telco_data.TotalCharges[(telco_data["Churn"] == 0) ],
                color="Red", fill = True)
Tot = sns.kdeplot(telco_data.TotalCharges[(telco_data["Churn"] == 1) ],
                ax =Tot, color="Blue", shade= True)
Tot.legend(["No Churn","Churn"],loc='upper right')
Tot.set_ylabel('Density')
Tot.set_xlabel('Total Charges')
Tot.set_title('Total charges by churn')
print(Tot)

# Build a corelation of all predictors with 'Churn'
plt.figure(figsize=(20,8))
telco_data.corr()['Churn'].sort_values(ascending = False).plot(kind='bar')

plt.figure(figsize=(12,12))
sns.heatmap(telco_data.corr(),cmap="Paired")

new_df1_target0=telco_data.loc[telco_data["Churn"]==0]
new_df1_target1=telco_data.loc[telco_data["Churn"]==1]

# 12. Dependent and Independent variables
X=telco_data.drop('Churn',axis=1)
y=telco_data['Churn']

X.head()
y.head()

# 13. split the data into training and testing
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

# 14.Import Algorithm
from sklearn.tree import DecisionTreeClassifier
model_dt =  DecisionTreeClassifier(criterion = "gini",random_state = 100,max_depth=6, min_samples_leaf=8)
model_dt.fit(X_train,y_train)

# 14.1 accuracy score
model_dt.score(X_test,y_test)

y_pred = model_dt.predict(X_test)
print(y_pred[:10])
print(y_test[:10])

# 14.2 Classification report
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred, labels=[0,1]))

# 14.3 confusion matrix
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test,y_pred))

# accuracy is quite low, and as it's an imbalanced dataset, we shouldn't consider Accuracy as our metrics to measure the model, as Accuracy is cursed in imbalanced datasets.
# Hence, we need to check recall, precision & f1 score for the minority class, and it's quite evident that the precision, recall & f1 score is too low for Class 1, i.e. churned customers.
# Hence, moving ahead to call SMOTEENN (UpSampling + ENN)
# main advantage of using SMOTEENN is that it addresses both overfitting and underfitting issues that can arise from class imbalance. 
# 15. By generating synthetic samples and removing noisy ones

from imblearn.over_sampling import SMOTE
smote = SMOTE()
X_ovs,y_ovs=smote.fit_resample(X,y)
fig, oversp = plt.subplots()
oversp.pie( y_ovs.value_counts(), autopct='%.2f')
oversp.set_title("Over-sampling")
plt.show()

# 16. split the sampled data into training and testing
Xr_train,Xr_test,yr_train,yr_test=train_test_split(X_ovs, y_ovs,test_size=0.2,random_state=42)

print("====================================================================")

# 17. Import Logistic Regression
from sklearn.linear_model import LogisticRegression

# 17.1 Define model as object
model_lr = LogisticRegression(max_iter=2000)
model_lr.fit(Xr_train, yr_train)

# 17.2 Define y_pred_lr variable (Logistic Regression Predictions)
y_pred_lr = model_lr.predict(Xr_test)

# Show first 10 predictions and actual values
print("Logistic Regression - First 10 Predictions:", y_pred_lr[:10])
print("Logistic Regression - First 10 True Labels:", yr_test[:10])

# 17.3 Classification Report for Logistic Regression
from sklearn.metrics import classification_report
class_report_lr = classification_report(yr_test, y_pred_lr)
print("Classification Report for Logistic Regression:\n", class_report_lr)

# 17.4 Accuracy Score for Logistic Regression
from sklearn.metrics import accuracy_score
acc_score_lr = accuracy_score(yr_test, y_pred_lr)
print("Accuracy Score for Logistic Regression:", acc_score_lr)

# 17.5 Confusion Matrix for Logistic Regression
from sklearn.metrics import confusion_matrix
con_matrix_lr = confusion_matrix(yr_test, y_pred_lr)
print("Confusion Matrix for Logistic Regression:\n", con_matrix_lr)

print("====================================================================")

# 18. Import Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier

# 18.1 Define model as object
model_dtc = DecisionTreeClassifier(criterion="gini", random_state=100, max_depth=6, min_samples_leaf=8)
model_dtc.fit(Xr_train, yr_train)

# 18.2 Define y_pred_dtc variable (Decision Tree Predictions)
y_pred_dtc = model_dtc.predict(Xr_test)

# Show first 10 predictions and actual values
print("Decision Tree - First 10 Predictions:", y_pred_dtc[:10])
print("Decision Tree - First 10 True Labels:", yr_test[:10])

# 18.3 Classification Report for Decision Tree Classifier
class_report_dtc = classification_report(yr_test, y_pred_dtc)
print("Classification Report for Decision Tree Classifier:\n", class_report_dtc)

# 18.4 Accuracy Score for Decision Tree Classifier
acc_score_dtc = accuracy_score(yr_test, y_pred_dtc)
print("Accuracy Score for Decision Tree Classifier:", acc_score_dtc)

# 18.5 Confusion Matrix for Decision Tree Classifier
con_matrix_dtc = confusion_matrix(yr_test, y_pred_dtc)
print("Confusion Matrix for Decision Tree Classifier:\n", con_matrix_dtc)

print("====================================================================")


# 19. Import RandomForest Classifier
from sklearn.ensemble import RandomForestClassifier

# 19.1 Define model as object
model_rfc = RandomForestClassifier(n_estimators=100, random_state=100, max_depth=6, min_samples_leaf=8, class_weight='balanced')
model_rfc.fit(Xr_train, yr_train)

# 19.2 Define y_pred_rf variable (RandomForest Predictions)
y_pred_rf = model_rfc.predict(Xr_test)

# Show first 10 predictions and actual values
print("RandomForest - First 10 Predictions:", y_pred_rf[:10])
print("RandomForest - First 10 True Labels:", yr_test[:10])

# 19.3 Accuracy Score for RandomForest Classifier
print("RandomForest Accuracy Score:", model_rfc.score(Xr_test, yr_test))

# 19.4 Classification Report for RandomForest Classifier
report_rfc = classification_report(yr_test, y_pred_rf)
print("Classification Report for RandomForest Classifier:\n", report_rfc)

# 19.5 Confusion Matrix for RandomForest Classifier
from sklearn.metrics import confusion_matrix
con_matrix_rf = confusion_matrix(yr_test, y_pred_rf)
print("Confusion Matrix for RandomForest Classifier:\n", con_matrix_rf)

print("====================================================================")

# 20. Import AdaBoost Classifier
from sklearn.ensemble import AdaBoostClassifier

# 20.1 Define model as object
model_abc = AdaBoostClassifier(algorithm='SAMME',n_estimators=100)
model_abc.fit(Xr_train, yr_train)

# 20.2 Define y_pred_abc variable (AdaBoost Predictions)
y_pred_abc = model_abc.predict(Xr_test)

# Show first 10 predictions and actual values
print("AdaBoost - First 10 Predictions:", y_pred_abc[:10])
print("AdaBoost - First 10 True Labels:", yr_test[:10])

# 20.3 Classification Report for AdaBoost Classifier
report_abc = classification_report(yr_test, y_pred_abc)
print("Classification Report for AdaBoost Classifier:\n", report_abc)

# 20.4 Confusion Matrix for AdaBoost Classifier
con_matrix_abc = confusion_matrix(yr_test, y_pred_abc)
print("Confusion Matrix for AdaBoost Classifier:\n", con_matrix_abc)

print("====================================================================")

# 21. Import GradientBoost Classifier
from sklearn.ensemble import GradientBoostingClassifier

# 21.1 Define model as object
model_gbc = GradientBoostingClassifier()
model_gbc.fit(Xr_train, yr_train)

# 21.2 Define y_pred_gbc variable (GradientBoost Predictions)
y_pred_gbc = model_gbc.predict(Xr_test)

# Show first 10 predictions and actual values
print("GradientBoost - First 10 Predictions:", y_pred_gbc[:10])
print("GradientBoost - First 10 True Labels:", yr_test[:10])

# 21.3 Classification Report for GradientBoost Classifier
report_gbc = classification_report(yr_test, y_pred_gbc)
print("Classification Report for GradientBoost Classifier:\n", report_gbc)

# 21.4 Confusion Matrix for GradientBoost Classifier
con_matrix_gbc = confusion_matrix(yr_test, y_pred_gbc)
print("Confusion Matrix for GradientBoost Classifier:\n", con_matrix_gbc)

print("====================================================================")

# 22. Import XGBoost Classifier
from xgboost import XGBClassifier

# 22.1 Define model as object
model_xgb = XGBClassifier(scale_pos_weight=len(yr_train[yr_train == 0]) / len(yr_train[yr_train == 1]))
model_xgb.fit(Xr_train, yr_train)

# 22.2 Define y_pred_xgb variable (XGBoost Predictions)
y_pred_xgb = model_xgb.predict(Xr_test)

# Show first 10 predictions and actual values
print("XGBoost - First 10 Predictions:", y_pred_xgb[:10])
print("XGBoost - First 10 True Labels:", yr_test[:10])

# 22.3 Classification Report for XGBoost Classifier
report_xgb = classification_report(yr_test, y_pred_xgb)
print("Classification Report for XGBoost Classifier:\n", report_xgb)

# 22.4 Confusion Matrix for XGBoost Classifier
con_matrix_xgb = confusion_matrix(yr_test, y_pred_xgb)
print("Confusion Matrix for XGBoost Classifier:\n", con_matrix_xgb)

print("====================================================================")


# Import necessary libraries
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix
import time

# Define your GradientBoostingClassifier model
model = GradientBoostingClassifier()

# Define the hyperparameter search space for RandomizedSearchCV
param_dist = {
    'learning_rate': [0.1, 0.5, 1.0],  # Testing different learning rates
    'n_estimators': [50, 100, 200],    # Number of boosting stages (trees) to test
    'max_depth': [3, 5, 7],            # Maximum depth of each tree
    'min_samples_split': [2, 5, 10]    # Minimum number of samples required to split an internal node
}

# Create a RandomizedSearchCV object with 5 iterations, 10-fold cross-validation, and accuracy scoring
random_search = RandomizedSearchCV(estimator=model, param_distributions=param_dist, n_iter=5, cv=10, 
                                   scoring='accuracy', random_state=42)

# Start the timer to track how long RandomizedSearchCV takes to run
start_time = time.time()

# Fit RandomizedSearchCV on the training data (Xr_train, yr_train)
random_search.fit(Xr_train, yr_train)

# Stop the timer after fitting
end_time = time.time()

# Calculate the total time taken for RandomizedSearchCV
total_time = end_time - start_time
print("RandomizedSearchCV took {:.2f} seconds to complete.".format(total_time))

# Get the best hyperparameters from the random search
best_params = random_search.best_params_
print("Best Parameters:", best_params)

# Define the best hyperparameters obtained from RandomizedSearchCV
# These are the hyperparameters that yielded the highest accuracy during cross-validation
best_params = {
    'n_estimators': 100,          # Best number of boosting stages
    'min_samples_split': 5,       # Best minimum number of samples to split a node
    'max_depth': 7,               # Best tree depth
    'learning_rate': 0.1          # Best learning rate for gradient boosting
}

# Create the final GradientBoostingClassifier with the best hyperparameters
final_gb_classifier = GradientBoostingClassifier(**best_params)

# Train the final model using the entire training set
final_gb_classifier.fit(Xr_train, yr_train)

# Perform 10-fold cross-validation to evaluate the trained model's performance
# cv_scores will contain the accuracy scores for each of the 10 folds
cv_scores = cross_val_score(final_gb_classifier, Xr_train, yr_train, cv=10, scoring='accuracy')

# Print the cross-validation scores and their mean
print("Cross-validation scores:", cv_scores)
print("Mean CV score:", cv_scores.mean())

# Use the trained model to make predictions on the test data (Xr_test)
y_pred_final = final_gb_classifier.predict(Xr_test)

# Show the first 10 predictions and the corresponding true labels for comparison
print("First 10 predictions:", y_pred_final[:10])
print("First 10 true labels:", yr_test[:10])

# Generate a classification report to summarize the model's precision, recall, f1-score, and support
print("\nClassification Report:\n", classification_report(y_pred_final, yr_test))

# Create a confusion matrix to visualize the number of correct and incorrect predictions
print("\nConfusion Matrix:\n", confusion_matrix(y_pred_final, yr_test))


import os 
import pickle
from sklearn.ensemble import GradientBoostingClassifier

# Change directory if needed
os.chdir(r'C:\Users\91939\Desktop\AI&DS\Data science projects\Project34_Customerchurn')

# Assuming final_gb_classifier is your trained model
# Define and train Gradient Boosting Classifier
best_params = {
    'n_estimators': 100,
    'min_samples_split': 5,
    'max_depth': 7,
    'learning_rate': 0.1
}

final_gb_classifier = GradientBoostingClassifier(**best_params)

# Train the final model on the entire training data (assuming Xr_train and yr_train are defined)
final_gb_classifier.fit(X_train, y_train)

# Dumping the model to a file
with open('final_gb_classifier.pkl', 'wb') as file:
    pickle.dump(final_gb_classifier, file)

# Load the saved model
with open('final_gb_classifier.pkl', 'rb') as file:
    loaded_model = pickle.load(file)



import pickle
import pandas as pd

# Load the saved model from the pickle file
with open('final_gb_classifier.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

# Prepare your own data for testing
# Create a DataFrame with your feature data
your_features = pd.DataFrame({
    'gender': [1, 0, 0, 0, 0],
    'SeniorCitizen': [0, 0, 0, 0, 0],
    'Partner': [0, 0, 0, 1, 1],
    'Dependents': [0, 0, 0, 0, 1],
    'PhoneService': [1, 0, 1, 1, 1],
    'MultipleLines': [0, 0, 0, 2, 2],
    'InternetService': [1, 0, 1, 1, 0],
    'OnlineSecurity': [0, 0, 0, 2, 2],
    'OnlineBackup': [0, 0, 1, 2, 2],
    'DeviceProtection': [0, 0, 0, 0, 2],
    'TechSupport': [0, 0, 0, 2, 2],
    'StreamingTV': [0, 1, 0, 0, 0],
    'StreamingMovies': [0, 1, 0, 0, 0],
    'Contract': [2, 0, 0, 1, 2],
    'PaperlessBilling': [0, 1, 0, 0, 0],
    'PaymentMethod': [1, 1, 1, 0, 0],
    'MonthlyCharges': [90.407734, 58.273891, 74.379767, 108.55, 64.35],
    'TotalCharges': [707.535237, 3264.466697, 1146.937795, 5610.7, 1558.65],
    'tenure_group': [0, 4, 1, 4, 2]
})

# Make predictions using the loaded model on your own data
predictions = loaded_model.predict(your_features)

# Print the predictions
print("Predictions:", predictions)
































































































