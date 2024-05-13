import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def read_and_process_data(Data):
    data = []
    labels = []

    for filename in os.listdir(Data):
        if filename.endswith(".txt"):
            label = filename.split("_")[0]  # Extract label from filename
            filepath = os.path.join(Data, filename)
            with open(filepath, 'r') as file:
                lines = file.readlines()
                for line in lines:
                    accel_data, _, gyro_data, _ = line.strip().split("  ")
                    accel_values = [float(val) for val in accel_data.split()[1:]]
                    gyro_values = [float(val) for val in gyro_data.split()[1:]]
                    data.append(accel_values + gyro_values)
                    labels.append(label)

    return data, labels

# Read and preprocess data
data, labels = read_and_process_data("Data")

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Initialize and train the Random Forest classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)

# Make predictions
y_pred = rf_classifier.predict(X_test)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

