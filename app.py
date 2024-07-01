from flask import Flask, request, jsonify
import pandas as pd
import openai
import joblib
import re
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

app = Flask(__name__)

# Load the model
try:
    model = joblib.load('delinquency_model_optimized2.pkl')
except ModuleNotFoundError:
    print("Error loading the model. Please ensure the correct scikit-learn version is installed.")
    raise

# Set up OpenAI API
openai.api_key = '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'

# Email configuration
SENDER_EMAIL = "ibrahimpuri65@gmail.com"
RECEIVER_EMAIL = "ibrahim.puri@netsoltech.com"
EMAIL_PASSWORD = "rdfk jngk qkxm kqxh"

# Function to create prompt
def create_prompt(data_row):
    prompt_template = """
    Generate a contract with the following details:
    - Loan Amount: {loan_amnt}
    - Outstanding Principal: {out_prncp}
    - Total Payment: {total_pymnt}1111
    - Total Received Interest: {total_rec_int}
    - Loan ID: {loan_id}
    - Borrower: {borrower}
    - Due Date: {due_date}
    - Status: {status}
    Fill in any missing details and ensure the contract is complete and accurate.
    """
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
    # Create the email content
    subject = "Urgent: Delinquent Contract Detected"
    body = f"A delinquent contract has been detected. Please review the following contract details urgently:\n\n{contract_details}\n\n{contract_data}"

    # Set up the MIME
    message = MIMEMultipart()
    message["From"] = SENDER_EMAIL
    message["To"] = RECEIVER_EMAIL
    message["Subject"] = subject
    message.attach(MIMEText(body, "plain"))

    # Send the email
    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(SENDER_EMAIL, EMAIL_PASSWORD)
        text = message.as_string()
        server.sendmail(SENDER_EMAIL, RECEIVER_EMAIL, text)
        server.quit()
        print("ALERT: Email sent successfully!")
    except Exception as e:
        print(f"Failed to send email: {e}")

@app.route('/process_contract', methods=['POST'])
def process_contract():
    data = request.json
    df = pd.DataFrame([data])

    for index, row in df.iterrows():
        prompt = create_prompt(row)
        contract = generate_contract(prompt)
        print(f"Generated Contract for Row {index}:\n{contract}\n")

        # Predict delinquency
        delinquency_prediction = predict_delinquency(contract)
        print(f"Delinquency Prediction for Row {index}: {delinquency_prediction}\n")

        # Send alert if contract is predicted to be delinquent
        if delinquency_prediction == 'delinquent':
            send_alert(contract, row.to_dict())

    return jsonify({"message": "Contract processed and email sent if delinquent."})

if __name__ == '__main__':
    app.run(debug=True)