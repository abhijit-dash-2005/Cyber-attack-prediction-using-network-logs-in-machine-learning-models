import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import numpy as np

# Folder path containing your multiple CSV files
folder_path = r"C:\Users\dasha\Desktop\ML PROJECTS\SUNIL SIR\PROJECT\datasets\MachineLearningCSV\MachineLearningCVE"

# List all CSV files in the folder
csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

# Load and concatenate all CSV files into one DataFrame
df_list = []
for file in csv_files:
    file_path = os.path.join(folder_path, file)
    df = pd.read_csv(file_path)
    df_list.append(df)

full_df = pd.concat(df_list, ignore_index=True)
print(f"Combined dataset shape: {full_df.shape}")

# Print columns to identify the label column
print("Columns in dataset:")
print(full_df.columns)

# Manually specify or find the name of the label column here after inspection
label_col = ' Label'  # Note the leading space

# Validate label column exists
if label_col not in full_df.columns:
    raise ValueError(f"Label column '{label_col}' not found in dataset columns.")

# Drop non-informative columns
cols_to_drop = ['Flow ID', 'Source IP', 'Destination IP', 'Timestamp']
for col in cols_to_drop:
    if col in full_df.columns:
        full_df.drop(columns=col, inplace=True)

# Handle infinite and missing values
full_df.replace([np.inf, -np.inf], np.nan, inplace=True)
full_df.dropna(inplace=True)

# Encode label column to numeric
label_encoder = LabelEncoder()
full_df[label_col] = label_encoder.fit_transform(full_df[label_col])

# Separate features and target
X = full_df.drop(columns=[label_col])
y = full_df[label_col]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Optional: convert scaled features back to DataFrame
X_preprocessed = pd.DataFrame(X_scaled, columns=X.columns)

print("Preprocessing complete.")
print("Features shape:", X_preprocessed.shape)
print("Labels shape:", y.shape)
print("Class labels:", label_encoder.classes_)

# Add labels back to scaled features DataFrame
X_preprocessed[label_col] = y.values

# Save to CSV file
X_preprocessed.to_csv('cicids2017_preprocessed 2.csv', index=False)
print("Preprocessed dataset saved as cicids2017_preprocessed.csv")
