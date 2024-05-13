import serial, time
dev = serial.Serial('COM14', timeout=0)

fp=open("./data.txt", 'w')
fp.write("acce_x,acce_y,acce_z,gyro_x,gyro_y,gyro_z\n")
i=0
print("start")
while i < 200:
    time.sleep(1/200)
    print(dev.readline())
fp.close()
print("stop")
#Model