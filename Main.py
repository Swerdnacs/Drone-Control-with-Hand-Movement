import serial, time
import djitellopy as Tello
import numpy as np
import joblib

import pandas as pd
from scipy.stats import skew, kurtosis
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Function to extract features from accelerometer and gyroscope data
def extract_features(file_path):
    # Load data from file
    data = pd.read_csv(file_path)
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

tello = Tello()
tello.connect()

tello.streamon()
print(tello.get_battery())
tello.turn_motor_on()
tello.takeoff()

dev = serial.Serial('COM12', timeout=0)
while True:
    fp=open("./data.txt", 'w')
    fp.write("acce_x,acce_y,acce_z,gyro_x,gyro_y,gyro_z\n")
    for i in range (1,200):
        time.sleep(1/100)
        tmp = dev.readline()
        if(tmp!=b''):
            fp.write(tmp[:-1].decode())
    fp.close()
    #Model
    # List to store features for all files
    all_features = []
    # List of gestures
    gestures = ["down", "up", "left", "right"]
    # List to store labels
    labels = []
    # Iterate over gestures and numbers to generate file paths
    for gesture_index, gesture in enumerate(gestures):
        for i in range(1, 51):  # Numbers from 1 to 50
            file_path = f"data.txt"
            features = extract_features(file_path)
            all_features.append(features)
            labels.append(gesture_index)  # Assign numerical labels to gestures
    # Convert features and labels to NumPy arrays
    X = np.array(all_features)
    y = np.array(labels)
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Initialize and train the Random Forest classifier
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_classifier.fit(X_train, y_train)
    # Evaluate the model
    accuracy = rf_classifier.score(X_test, y_test)
    print("Accuracy:", accuracy)
    # Read features from the mystery data
    file_path = "data.txt"
    pred_features = extract_features(file_path)
    # Convert the features to a numpy array
    pred_features_array = np.array([pred_features])
    # Use the trained classifier to predict the gesture
    predicted_gesture_index = rf_classifier.predict(pred_features_array)[0]
    # Map the predicted index to the actual gesture label
    predicted_gesture = gestures[predicted_gesture_index]
    print("Predicted Gesture:", predicted_gesture)
    #Drone
    match predicted_gesture:
        case 'down':
            tello.move_down(5)
        case 'up':
            tello.move_up(5)
        case 'right':
            tello.move_left(5)
        case 'left':
            tello.move_right(5)