import serial
import time

# Open serial connection to Arduino
ser = serial.Serial('/dev/ttyACM0', 115200, timeout=1)
time.sleep(2)  # Allow time for connection

while True:
    ser.write(b"Hello from Raspberry Pi\n")  # Send message to Arduino
    response = ser.readline().decode('utf-8').strip()  # Read response
    if response:
        print("Arduino says:", response)
    time.sleep(1)
