import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib

# Function to extract features from accelerometer and gyroscope data
def extract_features(file_path):
    # Load data from file
    data = pd.read_csv(file_path)
    
    # Check for missing values and handle them if necessary
    data.dropna(inplace=True)
    
    # columns are labeled 'acce_x', 'acce_y', 'acce_z', 'gyro_x', 'gyro_y', 'gyro_z'
    accelerometer_data = data[['acce_x', 'acce_y', 'acce_z']]
    gyroscope_data = data[['gyro_x', 'gyro_y', 'gyro_z']]
    
    # Initialize list to store features
    features = []
    
    # Extract statistical features for accelerometer data
    for axis in accelerometer_data.columns:
        features.append(np.mean(accelerometer_data[axis]))
        features.append(np.std(accelerometer_data[axis]))
        features.append(skew(accelerometer_data[axis]))
        features.append(kurtosis(accelerometer_data[axis]))
    
    # Extract statistical features for gyroscope data
    for axis in gyroscope_data.columns:
        features.append(np.mean(gyroscope_data[axis]))
        features.append(np.std(gyroscope_data[axis]))
        features.append(skew(gyroscope_data[axis]))
        features.append(kurtosis(gyroscope_data[axis]))
    
    return features

# List to store features for all files
all_features = []

# List of gestures
gestures = ["down", "up", "left", "right"]

# List to store labels
labels = []

# Iterate over gestures and numbers to generate file paths
for gesture_index, gesture in enumerate(gestures):
    for i in range(1, 51):  # Numbers from 1 to 50
        file_path = f"data/{gesture}_{str(i).zfill(2)}.txt" 
        try:
            features = extract_features(file_path)
            all_features.append(features)
            labels.append(gesture_index)  # Assign numerical labels to gestures
        except Exception as e:
            print(f"Error processing file {file_path}: {str(e)}")

# Convert features and labels to NumPy arrays
X = np.array(all_features)
y = np.array(labels)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define a pipeline with scaling and Random Forest classifier
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('rf', RandomForestClassifier(random_state=42))
])

# Define hyperparameters to search over
param_grid = {
    'rf__n_estimators': [50, 100, 150],
    'rf__max_depth': [None, 10, 20],
    'rf__min_samples_split': [2, 5, 10]
}

# Perform grid search cross-validation
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Get the best model from grid search
best_model = grid_search.best_estimator_

# Evaluate the best model
accuracy = best_model.score(X_test, y_test)
print("Accuracy:", accuracy)

# Save the trained model to a file
joblib.dump(best_model, 'new_model.joblib')

# Read features from a new right gesture data set
file_path = "prediction_data/move_left.txt" 
pred_features = extract_features(file_path)

# Convert the features to a numpy array
pred_features_array = np.array([pred_features])

# Use the trained classifier to predict the gesture
predicted_gesture_index = best_model.predict(pred_features_array)[0]

# Map the predicted index to the actual gesture label
predicted_gesture = gestures[predicted_gesture_index]

print("Predicted Gesture:", predicted_gesture)

