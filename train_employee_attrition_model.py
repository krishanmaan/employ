import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load the raw dataset
data = pd.read_csv('employee_attrition.csv')

# Step 2: Handle missing values
# Fill missing values in numeric columns only with the mean
numeric_columns = data.select_dtypes(include=["number"]).columns
data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].mean())

# Alternatively, you can drop rows with missing target values (Attrition)
data.dropna(subset=["Attrition"], inplace=True)

# Step 3: Encode categorical variables
# Identify categorical columns
categorical_columns = data.select_dtypes(include=["object"]).columns

label_encoder = LabelEncoder()
data["Attrition"] = label_encoder.fit_transform(data["Attrition"])

# Save the modified DataFrame back to the CSV file
data.to_csv('employee_attrition_processed.csv', index=False)

# Apply Label Encoding to other categorical features
for column in categorical_columns:
    if column != "Attrition":  # Skip the target column
        # Encode 'Yes'/'No' columns to 1/0 (if they exist in your dataset)
        if data[column].dtype == 'object' and data[column].isin(['Yes', 'No']).all():
            data[column] = data[column].map({'Yes': 1, 'No': 0})
        else:
            # Use Label Encoding for other categorical columns
            data[column] = label_encoder.fit_transform(data[column])

# Step 4: Feature selection (choose your relevant features)
features = data[["Age", "MonthlyIncome", "JobSatisfaction", "WorkLifeBalance", 
                "YearsAtCompany", "JobInvolvement", "EnvironmentSatisfaction",
                "DistanceFromHome", "PerformanceRating", "StockOptionLevel"]]

# Step 5: Split the data into training and testing sets - Changed to 60-40 split
X_train, X_test, y_train, y_test = train_test_split(features, data["Attrition"], test_size=0.4, random_state=42)

# Step 6: Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save the scaler for use in the app
joblib.dump(scaler, "scaler_model.pkl")

# Step 7: Train a Random Forest model
rf_model = RandomForestClassifier(random_state=42, 
                               n_estimators=100,
                               max_depth=10)
rf_model.fit(X_train_scaled, y_train)

# Step 8: Evaluate the Random Forest model
rf_pred = rf_model.predict(X_test_scaled)
rf_accuracy = accuracy_score(y_test, rf_pred)
print(f"Random Forest Model Accuracy: {rf_accuracy:.2f}")
print("\nRandom Forest Classification Report:")
print(classification_report(y_test, rf_pred))

# Step 9: Train a Gradient Boosting model (additional algorithm)
gb_model = GradientBoostingClassifier(random_state=42,
                                     n_estimators=100,
                                     learning_rate=0.1,
                                     max_depth=5)
gb_model.fit(X_train_scaled, y_train)

# Step 10: Evaluate the Gradient Boosting model
gb_pred = gb_model.predict(X_test_scaled)
gb_accuracy = accuracy_score(y_test, gb_pred)
print(f"Gradient Boosting Model Accuracy: {gb_accuracy:.2f}")
print("\nGradient Boosting Classification Report:")
print(classification_report(y_test, gb_pred))

# Step 11: Save both models
joblib.dump(rf_model, "rf_attrition_model.pkl")
joblib.dump(gb_model, "gb_attrition_model.pkl")
print("Models saved as rf_attrition_model.pkl and gb_attrition_model.pkl")

# Step 12: Create and save feature importance plots for visualization in the app
plt.figure(figsize=(12, 6))
feat_importances_rf = pd.Series(rf_model.feature_importances_, index=features.columns)
feat_importances_rf.nlargest(10).plot(kind='barh')
plt.title('Random Forest Feature Importance')
plt.tight_layout()
plt.savefig('rf_feature_importance.png')

plt.figure(figsize=(12, 6))
feat_importances_gb = pd.Series(gb_model.feature_importances_, index=features.columns)
feat_importances_gb.nlargest(10).plot(kind='barh')
plt.title('Gradient Boosting Feature Importance')
plt.tight_layout()
plt.savefig('gb_feature_importance.png')
