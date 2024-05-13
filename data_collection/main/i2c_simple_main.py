import smbus
import time

# MPU6050 register addresses
MPU6050_ACCEL_XOUT_H = 0x3B
MPU6050_PWR_MGMT_1 = 0x6B

# I2C bus number
I2C_BUS = 1

# MPU6050 device address
MPU6050_ADDR = 0x68

# Function to initialize I2C
def i2c_init():
    return smbus.SMBus(I2C_BUS)

# Function to initialize MPU6050
def mpu6050_init(bus):
    # Wake up MPU6050
    bus.write_byte_data(MPU6050_ADDR, MPU6050_PWR_MGMT_1, 0)

# Function to read accelerometer data
def read_acceleration(bus):
    data = bus.read_i2c_block_data(MPU6050_ADDR, MPU6050_ACCEL_XOUT_H, 6)
    x = data[0] << 8 | data[1]
    y = data[2] << 8 | data[3]
    z = data[4] << 8 | data[5]
    return x, y, z

# Function to read gyroscope data
def read_gyroscope(bus):
    # Not implemented in this example
    pass

def main():
    # Initialize I2C
    bus = i2c_init()

    # Initialize MPU6050
    mpu6050_init(bus)

    print("acce_x,acce_y,acce_z")

    start_time = time.time()
    duration = 2

    while (time.time() - start_time) < duration:
        # Read accelerometer data
        x, y, z = read_acceleration(bus)

        # Print data
        print(f"{x},{y},{z}")

        time.sleep(0.1)  # Adjust delay as needed

if __name__ == "__main__":
    main()
