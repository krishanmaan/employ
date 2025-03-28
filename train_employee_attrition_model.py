import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score
import joblib

# Step 1: Load the raw dataset (replace with your actual data source)
data = pd.read_csv('C:\\Users\\aa\\Desktop\\MCA_Project\\employee_attrition.csv')

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
data.to_csv('C:\\Users\\aa\\Desktop\\MCA_Project\\employee_attrition.csv', index=False)

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
features = data[["Age", "MonthlyIncome", "JobSatisfaction", "WorkLifeBalance", "YearsAtCompany", "JobInvolvement", "EnvironmentSatisfaction"]]

# Step 5: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, data["Attrition"], test_size=0.2, random_state=42)

# Step 6: Feature scaling (optional but recommended for some algorithms)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 7: Train a Random Forest model
model = RandomForestClassifier(random_state=42)
model.fit(X_train_scaled, y_train)

# Step 8: Evaluate the model
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# Step 9: Save the model to a .pkl file
joblib.dump(model, "employee_attrition_model.pkl")
print("Model saved as employee_attrition_model.pkl")
