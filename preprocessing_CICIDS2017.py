import os
os.makedirs('datasets',exist_ok=True)
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# Ensure the output folder exists
os.makedirs('datasets', exist_ok=True)

# Load the dataset
df = pd.read_csv("C:\\Users\\dasha\\Desktop\\ML PROJECTS\\SUNIL SIR\\PROJECT\\CICIDS2017\\cicids2017_cleaned.csv")

# Clean: replace inf/-inf, drop NaN
df = df.replace([float('inf'), float('-inf')], pd.NA).dropna()

# Encode categorical columns
cat_cols = df.select_dtypes(include=['object']).columns
for col in cat_cols:
    df[col] = LabelEncoder().fit_transform(df[col])

# Normalize numeric columns
num_cols = df.select_dtypes(include=['int64', 'float64']).columns
df[num_cols] = MinMaxScaler().fit_transform(df[num_cols])

# Save the preprocessed dataset
df.to_csv(r"C:\Users\dasha\Desktop\ML PROJECTS\SUNIL SIR\PROJECT\datasets\CICIDS2017_preprocessed.csv",index=False)

