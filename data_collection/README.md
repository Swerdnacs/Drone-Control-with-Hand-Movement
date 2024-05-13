
# This homework uses i2c_simple as a base example for the assignment

## Overview

This example demonstrates reading acceleration and gyroscope data from an MPU6050 sensor over I2C connection using the espressif MPU6050 driver

### Hardware Required

To run this example, use ESP32-S3 and MPU6050.

#### Pin Assignment:

**Note:** The following pin assignments are used by default, you can change these in the `menuconfig` .

|                  | SDA             | SCL           |
| ---------------- | -------------- | -------------- |
| ESP I2C Master   | I2C_MASTER_SDA | I2C_MASTER_SCL |
| MPU6050 Sensor   | SDA            | SCL            |


For the actual default value of `I2C_MASTER_SDA` and `I2C_MASTER_SCL` see `Example Configuration` in `menuconfig`.

Quick refrence: VCC-> 5V GND-> G SCL-> 1 SDA-> 0

**Note:** There's no need to add an external pull-up resistors for SDA/SCL pin, because the driver will enable the internal pull-up resistors.

### Build and Flash

Enter `idf.py flash monitor` to build, flash and monitor the project.

To exit, do CTRL + X

See the [Getting Started Guide](https://docs.espressif.com/projects/esp-idf/en/latest/get-started/index.html) for full steps to configure and use ESP-IDF to build projects.

## Example Output

```bash
I (328) i2c-simple-example: I2C initialized successfully
acce_x,acce_y,acce_z,gyro_x,gyro_y,gyro_z
0.248047	-0.375977	0.769287	-1.473282	-4.78626	-3.687023
0.239746	-0.380615	0.784424	-2	-5.236641	-4.40458
0.245605	-0.376465	0.787598	-1.694656	-5.351145	-4.923664
0.250488	-0.38501	0.788086	-0.78626	-6.114504	-6.900764
...
```

# Plot Analysis

Copy and paste the termial output to corresponding csv files. Then, format using tools like Google Sheets. Save the data to sensor_data. 

Run imageGen in MatLab to generate and save the produced images to the images folder within the zip file. 
