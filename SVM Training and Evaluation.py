import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
import joblib

# Load dataset
df = pd.read_csv(r"C:\Users\dasha\Desktop\ML PROJECTS\SUNIL SIR\PROJECT\datasets\cicids2017_feature_selected 2.csv")

# Clean column names
df.columns = df.columns.str.strip()

# Label column name (update if necessary)
label_column = 'Label'  # Change to 'Label' if it's 'Attack Type' mistakenly used

# Split features and labels
X = df.drop(label_column, axis=1)
y = df[label_column]

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Scale features to [0,1] range
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Stratified train/test split (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
)

# Initialize LinearSVC
svm_model = LinearSVC(class_weight="balanced", random_state=42, max_iter=5000, dual=False)
svm_model.fit(X_train, y_train)

# Predict and evaluate
y_pred = svm_model.predict(X_test)

print("LinearSVC Accuracy:", accuracy_score(y_test, y_pred))
print("LinearSVC F1 Score (macro):", f1_score(y_test, y_pred, average='macro'))
print("Classification Report:\n", classification_report(y_test, y_pred, target_names=[str(c) for c in le.classes_]))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Save the model
joblib.dump(svm_model, r"C:\Users\dasha\Desktop\ML PROJECTS\SUNIL SIR\PROJECT\Models\svm_model 2.pkl")
print("Model saved as 'svm_model 2.pkl'")
