import pandas as pd

# Load the data
data_path = 'loan_data.csv'  # Update with your actual path
data = pd.read_csv(data_path)

# Inspect the data
print("Data loaded successfully.")
print("First few rows of the data:")
print(data.head())

# Display basic information about the data
print("\nData Information:")
print(data.info())

# Check for duplicates
print("\nNumber of duplicate rows:", data.duplicated().sum())

# Show duplicate rows if any
print("\nDuplicate rows:")
print(data[data.duplicated()])

# Remove duplicate rows
data = data.drop_duplicates()
print("\nDuplicates removed.")
print("\nNumber of remaining rows:", len(data))

# Filter out rows with non-positive values in selected columns
columns_to_check = ['loan_amnt', 'out_prncp', 'total_pymnt', 'total_rec_prncp']
for col in columns_to_check:
    data = data[data[col] > 0]

print("\nFiltered out rows with non-positive values in selected columns.")
print("\nNumber of remaining rows:", len(data))

# Further filtering based on domain-specific knowledge
# Example: removing rows where 'total_pymnt' is less than 'loan_amnt'
data = data[data['total_pymnt'] >= data['loan_amnt']]
print("\nFiltered out rows where 'total_pymnt' is less than 'loan_amnt'.")
print("\nNumber of remaining rows:", len(data))

# Save the cleaned data
cleaned_data_path = 'cleaned_data.csv'  # Update with your desired path
data.to_csv(cleaned_data_path, index=False)
print(f"Cleaned data saved to {cleaned_data_path}")