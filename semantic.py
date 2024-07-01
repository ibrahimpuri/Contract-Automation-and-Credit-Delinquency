import pandas as pd

# Load the dataset
df = pd.read_csv('processed_data.csv')

# Display the first few rows of the dataset
print(df.head())

# Display the column names
print(df.columns.tolist())