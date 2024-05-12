import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis

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

# List to store features for all files
all_features = []

# List of gestures
gestures = ["down", "up", "left", "right"]

# Iterate over gestures and numbers to generate file paths
for gesture in gestures:
    for i in range(1, 51):  # Numbers from 1 to 50
        file_path = f"data/{gesture}_{str(i).zfill(2)}.txt"  # Assuming file extension is '.txt'
        features = extract_features(file_path)
        all_features.append(features)

# Convert features to NumPy array
all_features_array = np.array(all_features)

# Print shape of the feature matrix
print("Shape of feature matrix:", all_features_array.shape)

