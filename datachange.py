import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score
import joblib

# Step 1: Load the raw dataset
data = pd.read_csv('C:\\Users\\aa\\Desktop\\MCA_Project\\employee_attrition.csv')

# Step 2: Invert the values of the 'Attrition' column (swap 1 with 0 and vice versa)
data["Attrition"] = 1 - data["Attrition"]

# Save the modified DataFrame back to the CSV file (optional)
data.to_csv('C:\\Users\\aa\\Desktop\\MCA_Project\\employee_attrition.csv', index=False)

# Step 3: Handle missing values and encode categorical variables
numeric_columns = data.select_dtypes(include=["number"]).columns
data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].mean())
data.dropna(subset=["Attrition"], inplace=True)

categorical_columns = data.select_dtypes(include=["object"]).columns
label_encoder = LabelEncoder()
data["Attrition"] = label_encoder.fit_transform(data["Attrition"])

# Apply Label Encoding to other categorical features
for column in categorical_columns:
    if column != "Attrition":
        if data[column].dtype == 'object' and data[column].isin(['Yes', 'No']).all():
            data[column] = data[column].map({'Yes': 1, 'No': 0})
        else:
            data[column] = label_encoder.fit_transform(data[column])

# Step 4: Feature selection
features = data[["Age", "MonthlyIncome", "JobSatisfaction", "WorkLifeBalance", "YearsAtCompany", "JobInvolvement", "EnvironmentSatisfaction"]]

# Step 5: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, data["Attrition"], test_size=0.2, random_state=42)

# Step 6: Feature scaling
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
