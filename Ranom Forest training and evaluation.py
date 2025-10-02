import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import joblib

# Load the dataset with selected features
df = pd.read_csv(r"C:\Users\dasha\Desktop\ML PROJECTS\SUNIL SIR\PROJECT\datasets\cicids2017_feature_selected 2.csv")

# Clean column names for consistency
df.columns = df.columns.str.strip()

# Specify label column (use correct name from your CSV)
label_column = 'Label'  # Change to 'Label' as per your selected CSV, not 'Attack Type'

# Split into features and target
X = df.drop(label_column, axis=1)
y = df[label_column]

# Split data into training and testing sets (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Initialize Random Forest classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

# Train the model
clf.fit(X_train, y_train)

# Predict on test data
y_pred = clf.predict(X_test)

# Evaluate and print metrics
print("Accuracy:", accuracy_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred, average='macro'))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Save the trained model
joblib.dump(clf, 'RandomForest_model_2.pkl')
print("Model saved as 'RandomForest_model_2.pkl'")
