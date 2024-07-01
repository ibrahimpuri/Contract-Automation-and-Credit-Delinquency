import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import pandas as pd
import joblib
from datetime import datetime

# Load the model
model_path = 'delinquency_model_optimized.pkl'  # Update with your model path
model = joblib.load(model_path)

# Define a new sample input with contract details
new_input = pd.DataFrame({
    'loan_amnt': [15000],
    'out_prncp': [5000],
    'total_pymnt': [10000],
    'total_rec_prncp': [8000],
    'loan_to_income_ratio': [15000 / (10000 + 1)],
    'total_pymnt_perc': [10000 / (15000 + 1)],
    'principal_paid_ratio': [8000 / (15000 + 1)],
    'contract_id': ['C12345'],  # Example contract ID
    'customer_name': ['John Doe'],  # Example customer name
    'due_date': ['2024-12-31'],  # Example due date
    'email': ['john.doe@example.com']  # Example customer email
})

# Make a prediction
prediction = model.predict(new_input.drop(columns=['contract_id', 'customer_name', 'due_date', 'email']))

# Get the current timestamp
timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

# Determine the alert message
if prediction[0] == 0:
    alert_message = "Alert: This account is likely to be delinquent. Immediate action required."
else:
    alert_message = "This account is not likely to be delinquent."

# Print the alert
print(f"{timestamp} - {alert_message}")

# Function to send an email
def send_email(subject, body, to_email):
    from_email = "ibrahimpuri65@gmail.com"  # Replace with your Gmail address
    password = "rdfk jngk qkxm kqxh"
    # Create the email
    msg = MIMEMultipart()
    msg['From'] = from_email
    msg['To'] = to_email
    msg['Subject'] = subject

    msg.attach(MIMEText(body, 'plain'))

    try:
        # Connect to the Gmail SMTP server
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()  # Secure the connection
        server.login(from_email, password)
        server.sendmail(from_email, to_email, msg.as_string())
        server.quit()
        print("Email sent successfully.")
    except Exception as e:
        print(f"Failed to send email: {e}")

# Define email content
subject = "Delinquency Alert for Contract"
body = f"""
Dear Team,

{alert_message}

Contract Details:
- Contract ID: {new_input['contract_id'][0]}
- Customer Name: {new_input['customer_name'][0]}
- Due Date: {new_input['due_date'][0]}
- Email: {new_input['email'][0]}

This analysis was performed on {timestamp}.

Please act urgently on this contract if it is delinquent.

Best Regards,
Your Automated System
"""

# Send the email
to_email = "ibrahim.puri@netsoltech.com"  # Replace with your team's email address
send_email(subject, body, to_email)