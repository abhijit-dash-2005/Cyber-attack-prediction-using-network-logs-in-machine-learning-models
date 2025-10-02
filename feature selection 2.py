import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import MinMaxScaler
import joblib

# Path to your preprocessed dataset CSV file
preprocessed_csv = r"C:\Users\dasha\Desktop\ML PROJECTS\SUNIL SIR\PROJECT\datasets\cicids2017_preprocessed 2.csv"

# Load preprocessed dataset
df = pd.read_csv(preprocessed_csv)
df.columns = df.columns.str.strip()
label_col = 'Label'
X = df.drop(columns=[label_col])
y = df[label_col]

# Use MinMaxScaler to make all values non-negative
scaler = MinMaxScaler()
X_nonneg = scaler.fit_transform(X)
X_nonneg_df = pd.DataFrame(X_nonneg, columns=X.columns)

# Perform Chi-square feature selection
selector = SelectKBest(score_func=chi2, k=20)
X_selected = selector.fit_transform(X_nonneg_df, y)

# Get names of selected features
selected_features = X_nonneg_df.columns[selector.get_support()]
print("Selected features:", list(selected_features))

# Create DataFrame with selected features plus label
df_selected = pd.DataFrame(X_selected, columns=selected_features)
df_selected[label_col] = y.values

# Save feature-selected dataset
df_selected.to_csv('cicids2017_feature_selected 2.csv', index=False)
print("Feature-selected and scaled dataset saved as cicids2017_feature_selected 2.csv.")

# Save selector and scaler for future use
joblib.dump(selector, 'feature_selector.pkl')
joblib.dump(scaler, 'minmax_scaler.pkl')
