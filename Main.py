import serial, time
from djitellopy import Tello
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


tello = Tello()
tello.connect()

tello.streamon()
print(tello.get_battery())
tello.takeoff() 

dev = serial.Serial('COM12', timeout=0)
w=0
while w <7:
    fp=open("./data.txt", 'w')
    fp.write("acce_x,acce_y,acce_z,gyro_x,gyro_y,gyro_z\n")
    i=0
    print("start")
    while i < 300:
        time.sleep(1/201)
        tmp = dev.readline()
        if(tmp!=b''):
            fp.write(tmp[:-1].decode())
            i+=1
    fp.close()
    print("stop")
    #Model
    # Read features from a new right gesture data set
    file_path = "data.txt" 
    pred_features = extract_features(file_path)

    # Convert the features to a numpy array
    pred_features_array = np.array([pred_features])

    # Use the trained classifier to predict the gesture
    loaded_model = joblib.load('new_model.joblib')
    predicted_gesture_index = loaded_model.predict(pred_features_array)[0]

    # Map the predicted index to the actual gesture label
    gestures = ["down", "up", "left", "right"]
    predicted_gesture = gestures[predicted_gesture_index]

    print("Predicted Gesture:", predicted_gesture)

    #Drone
    match predicted_gesture:
        case 'down':
            tello.move_left(30)
        case 'up':
            tello.move_right(30)
        case 'right':
            tello.move_right(40)
        case 'left':
            tello.move_left(40)
    w+=1
    time.sleep(1)
tello.land()