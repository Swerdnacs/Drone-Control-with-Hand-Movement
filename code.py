import time
import board
import adafruit_mpu6050
import busio


i2c = busio.I2C(sda=board.IO2,scl=board.IO1)
# i2c = board.I2C()  # uses board.SCL and board.SDA
mpu = adafruit_mpu6050.MPU6050(i2c)
freq = 20
while True:
    a= str(mpu.acceleration[0])+','+str(mpu.acceleration[1])+','+str(mpu.acceleration[2])+','+str(mpu.gyro[0])+','+str(mpu.gyro[1])+','+str(mpu.gyro[2])
    print(a)
    time.sleep(1/100)
