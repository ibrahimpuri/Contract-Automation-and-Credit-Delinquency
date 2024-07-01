import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load and clean data
data_path = 'processed_data.csv'  # Update with the correct path
data = pd.read_csv(data_path)

# Check for any categorical features
categorical_features = data.select_dtypes(include=['object']).columns.tolist()

# Encode categorical data
for column in categorical_features:
    if column != 'delinquency_status':  # Ensure target variable is not encoded
        data[column] = data[column].astype('category').cat.codes

# Feature engineering
data['loan_to_income_ratio'] = data['loan_amnt'] / (data['total_pymnt'] + 1)
data['total_pymnt_perc'] = data['total_pymnt'] / (data['loan_amnt'] + 1)
data['principal_paid_ratio'] = data['total_rec_prncp'] / (data['loan_amnt'] + 1)
data['interest_paid'] = data['total_rec_int']

# Drop non-informative columns
features_to_drop = ['loan_id', 'borrower', 'due_date']
data.drop(columns=features_to_drop, inplace=True)

# Separate features and target
X = data.drop(columns='delinquency_status')
y = data['delinquency_status']

# Ensure all features are numeric
print("Feature types after encoding:")
print(X.dtypes)

# Apply SMOTE to balance the classes
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)
print("Class distribution after SMOTE:")
print(pd.Series(y_resampled).value_counts())

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled)

# Model definitions
models = {
    'RandomForest': RandomForestClassifier(random_state=42),
    'GradientBoosting': GradientBoostingClassifier(random_state=42),
    'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'),
    'SVM': SVC(kernel='linear', class_weight='balanced', random_state=42),
    'XGBoost': XGBClassifier(scale_pos_weight=(y_train.value_counts()[0] / y_train.value_counts()[1]), random_state=42)
}

# Evaluate models
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{name} accuracy: {accuracy * 100:.2f}%")
    print(classification_report(y_test, y_pred))

# Ensemble method
voting_clf = VotingClassifier(estimators=[
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
    ('gb', GradientBoostingClassifier(n_estimators=100, random_state=42)),
    ('xgb', XGBClassifier(scale_pos_weight=(y_train.value_counts()[0] / y_train.value_counts()[1]), random_state=42))
], voting='soft', n_jobs=-1)

voting_clf.fit(X_train, y_train)
y_pred_voting = voting_clf.predict(X_test)
accuracy_voting = accuracy_score(y_test, y_pred_voting)
print(f"Ensemble accuracy: {accuracy_voting * 100:.2f}%")
print(classification_report(y_test, y_pred_voting))

# Hyperparameter tuning
param_grid = {
    'n_estimators': [100, 200],
    'learning_rate': [0.01, 0.1],
    'max_depth': [3, 5]
}

grid_search = GridSearchCV(GradientBoostingClassifier(random_state=42), param_grid, cv=StratifiedKFold(n_splits=3), scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best cross-validation score: {grid_search.best_score_ * 100:.2f}%")

best_model = GradientBoostingClassifier(**grid_search.best_params_, random_state=42)
best_model.fit(X_train, y_train)
y_pred_best = best_model.predict(X_test)
accuracy_best = accuracy_score(y_test, y_pred_best)
print(f"Optimized model accuracy: {accuracy_best * 100:.2f}%")
print(classification_report(y_test, y_pred_best))

# Cross-validation
cv_scores = cross_val_score(best_model, X_resampled, y_resampled, cv=StratifiedKFold(n_splits=5), scoring='accuracy', n_jobs=-1)
print(f"Cross-validation scores: {cv_scores}")
print(f"Mean cross-validation score: {np.mean(cv_scores) * 100:.2f}%")

# Save the model
model_path = 'delinquency_model_optimized2.pkl'  # Update with your desired path
joblib.dump(best_model, model_path)
print(f"Model saved to {model_path}")

# Load the model for prediction
model = joblib.load(model_path)

# Define a new sample input
new_input = pd.DataFrame({
    'loan_amnt': [15000],
    'out_prncp': [5000],
    'total_pymnt': [10000],
    'total_rec_prncp': [8000],
    'loan_to_income_ratio': [15000 / (10000 + 1)],
    'total_pymnt_perc': [10000 / (15000 + 1)],
    'principal_paid_ratio': [8000 / (15000 + 1)]
})

# Make a prediction
prediction = model.predict(new_input)

# Print an alert based on the prediction
if prediction[0] == 0:
    print("Alert: This account is likely to be delinquent.")
else:
    print("This account is not likely to be delinquent.")