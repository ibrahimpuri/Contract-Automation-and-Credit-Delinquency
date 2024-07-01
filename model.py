import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load the cleaned data
cleaned_data_path = 'loan_data.csv'  # Update with the actual path
data = pd.read_csv(cleaned_data_path)

# Ensure consistent indentation
def determine_delinquency(status):
    if status.lower() == 'unpaid':
        return 0  # Unpaid is 0
    elif status.lower() == 'paid':
        return 1  # Paid is 1
    else:
        return -1  # Use -1 or another indicator for unknown status

data['delinquency_status'] = data['status'].apply(determine_delinquency)

print("Class distribution in 'delinquency_status':")
print(data['delinquency_status'].value_counts())

processed_data_path = 'processed_data.csv'  # Update with your desired path
data.to_csv(processed_data_path, index=False)
print(f"Processed data saved to {processed_data_path}")

# Ensure this line is properly aligned
features = ['loan_amnt', 'out_prncp', 'total_pymnt', 'total_rec_prncp']
target = 'delinquency_status'

# Ensure the lines below are indented correctly
X = data[features]
y = data[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {accuracy * 100:.2f}%")

model_path = 'delinquency_model.pkl'  # Update with your desired path
joblib.dump(model, model_path)
print(f"Model saved to {model_path}")

# Load the model for prediction
model = joblib.load(model_path)

# Define a new sample input
new_input = pd.DataFrame({
    'loan_amnt': [15000],
    'out_prncp': [5000],
    'total_pymnt': [10000],
    'total_rec_prncp': [8000]
})

# Make a prediction
prediction = model.predict(new_input)

# Print an alert based on the prediction
if prediction[0] == 0:
    print("Alert: This account is likely to be delinquent.")
else:
    print("This account is not likely to be delinquent.")