import pandas as pd
import openai
import joblib
import re
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

# Load the dataset
file_path = 'processed_data.csv'
data = pd.read_csv(file_path)

# Display the first few rows of the dataset to verify
print(data.head())

# Define important columns
important_columns = ['loan_amnt', 'out_prncp', 'total_pymnt', 'total_rec_int', 'loan_id', 'borrower', 'due_date', 'status', 'delinquency_status']

# Check if these columns are present in the dataset
for column in important_columns:
    if column not in data.columns:
        raise ValueError(f"Missing important column: {column}")

# Create a prompt template for the LLM
prompt_template = """
Generate a contract with the following details:
- Loan Amount: {loan_amnt}
- Outstanding Principal: {out_prncp}
- Total Payment: {total_pymnt}
- Total Received Interest: {total_rec_int}
- Loan ID: {loan_id}
- Borrower: {borrower}
- Due Date: {due_date}
- Status: {status}
Fill in any missing details and ensure the contract is complete and accurate.
"""

# Function to create prompt
def create_prompt(data_row):
    return prompt_template.format(
        loan_amnt=data_row['loan_amnt'],
        out_prncp=data_row['out_prncp'],
        total_pymnt=data_row['total_pymnt'],
        total_rec_int=data_row['total_rec_int'],
        loan_id=data_row['loan_id'],
        borrower=data_row['borrower'],
        due_date=data_row['due_date'],
        status=data_row['status']
    )

# Set up OpenAI API
openai.api_key = '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'

# Function to generate contract using GPT-3.5 Turbo
def generate_contract(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that generates contract documents."},
            {"role": "user", "content": prompt}
        ]
    )
    return response['choices'][0]['message']['content'].strip()

# Load the trained ML model
try:
    model = joblib.load('delinquency_model_optimized2.pkl')
except ModuleNotFoundError:
    print("Error loading the model. Please ensure the correct scikit-learn version is installed.")
    raise

# Function to extract features from contract using regular expressions
def extract_features_from_contract(contract_data):
    # Helper function to extract a value or return a default
    def extract_value(pattern, text, default):
        match = re.search(pattern, text)
        if match:
            return match.group(1).replace(',', '')
        else:
            return default

    loan_amnt = int(extract_value(r'Loan Amount: \$([\d,]+)', contract_data, '0'))
    out_prncp = float(extract_value(r'Outstanding Principal: \$([\d,\.]+)', contract_data, '0.0'))
    total_pymnt = float(extract_value(r'Total Payment: \$([\d,\.]+)', contract_data, '0.0'))
    total_rec_int = float(extract_value(r'Total Received Interest: \$([\d,\.]+)', contract_data, '0.0'))
    loan_id = int(extract_value(r'Loan ID: (\d+)', contract_data, '0'))
    status_paid = 1 if 'Status: Paid' in contract_data else 0

    # Example additional features (customize as necessary)
    num_terms_conditions = contract_data.count("Terms and Conditions")
    has_default_clause = 1 if "default" in contract_data else 0
    num_signatures = contract_data.count("Signature")

    features = [
        loan_amnt,
        out_prncp,
        total_pymnt,
        total_rec_int,
        loan_id,
        status_paid,
        len(contract_data),
        num_terms_conditions,
        has_default_clause,
        num_signatures
    ]
    return features

# Function to predict delinquency
def predict_delinquency(contract_data):
    features = extract_features_from_contract(contract_data)
    prediction = model.predict([features])
    return prediction[0]

# Function to send alert email
def send_alert(contract_data, contract_details):
    # Email configuration
    sender_email = "ibrahimpuri65@gmail.com"
    receiver_email = "ibrahim.puri@netsoltech.com"
    password = "rdfk jngk qkxm kqxh"

    # Create the email content
    subject = "Urgent: Delinquent Contract Detected"
    body = f"A delinquent contract has been detected. Please review the following contract details urgently:\n\n{contract_details}\n\n{contract_data}"

    # Set up the MIME
    message = MIMEMultipart()
    message["From"] = sender_email
    message["To"] = receiver_email
    message["Subject"] = subject
    message.attach(MIMEText(body, "plain"))

    # Send the email
    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(sender_email, password)
        text = message.as_string()
        server.sendmail(sender_email, receiver_email, text)
        server.quit()
        print("ALERT: Email sent successfully!")
    except Exception as e:
        print(f"Failed to send email: {e}")

# Number of rows to process
num_rows_to_process = 6

# Process the specified number of rows in the dataset
for index, row in data.head(num_rows_to_process).iterrows():
    prompt = create_prompt(row)
    contract = generate_contract(prompt)
    print(f"Generated Contract for Row {index}:\n{contract}\n")

    # Predict delinquency
    delinquency_prediction = predict_delinquency(contract)
    print(f"Delinquency Prediction for Row {index}: {delinquency_prediction}\n")

    # Send alert if contract is predicted to be delinquent
    if delinquency_prediction == 'delinquent':
        send_alert(contract, row)