"""
Quick CSV Cleaner - Fixes column name issues
"""
import pandas as pd

print("🔧 Cleaning loan_approval_dataset.csv...")

# Read with original column names
df = pd.read_csv('loan_approval_dataset.csv')

print(f"Original columns: {df.columns.tolist()}")

# Strip all whitespace from column names
df.columns = df.columns.str.strip()

print(f"Cleaned columns: {df.columns.tolist()}")

# Also strip whitespace from string values
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = df[col].str.strip()
        print(f"✅ Cleaned values in: {col}")

# Save cleaned version
df.to_csv('loan_approval_dataset_clean.csv', index=False)

print("\n✅ Cleaned file saved as: loan_approval_dataset_clean.csv")
print("✅ Now run: python train_advanced_models.py")