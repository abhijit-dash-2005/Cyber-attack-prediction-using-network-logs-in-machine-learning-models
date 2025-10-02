import pandas as pd

# Load dataset (replace with your file path)
df = pd.read_csv("C:\\Users\\dasha\\Desktop\\ML PROJECTS\\SUNIL SIR\\PROJECT\\datasets\\CICIDS2017_preprocessed.csv")
# Get number of rows and columns
rows, columns = df.shape

print(f"Number of rows: {rows}")
print(f"Number of columns: {columns}")
