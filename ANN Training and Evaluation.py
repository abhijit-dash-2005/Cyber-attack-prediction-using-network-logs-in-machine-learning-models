import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
import numpy as np

# Load dataset
df = pd.read_csv(r"C:\Users\dasha\Desktop\ML PROJECTS\SUNIL SIR\PROJECT\datasets\cicids2017_feature_selected 2.csv")

# Clean column names
df.columns = df.columns.str.strip()

# Label column - update if needed
label_column = 'Label'  # Change 'Attack Type' to 'Label' or actual column name

# Separate features and labels
X = df.drop(label_column, axis=1)
y = df[label_column]

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_categorical = to_categorical(y_encoded)

# Scale features to [0, 1]
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Train/test split with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_categorical, test_size=0.3, random_state=42, stratify=y_encoded
)

# Build ANN model
model = Sequential()
model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(y_categorical.shape[1], activation='softmax'))  # Multi-class output

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train model
model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=1)

# Predict on test data
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)
y_test_labels = np.argmax(y_test, axis=1)

# Evaluate
print("ANN Accuracy:", accuracy_score(y_test_labels, y_pred))
print("ANN F1 Score (macro):", f1_score(y_test_labels, y_pred, average='macro'))
print("Classification Report:\n", classification_report(y_test_labels, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test_labels, y_pred))

# Save the model
model.save(r"C:\Users\dasha\Desktop\ML PROJECTS\SUNIL SIR\PROJECT\Models\ANN_model_2.h5")
print("Model saved as 'ANN_model_2.h5'")
