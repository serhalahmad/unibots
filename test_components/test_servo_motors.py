import serial
import time

class Motor:
    def __init__(self, name, ser):
        self.name = name
        self.ser = ser
        # Initialize motor hardware here

    def setVelocity(self, left_velocity, right_velocity):
        # Send PWM signal or command to motor driver
        print(f"{self.name} for left wheels to {left_velocity} and right wheels to {right_velocity}")
        # Example: "setVelocity 40 75" (left motor 40, right motor 75)
        command = str(self.name) + " " + str(left_velocity) + " " + str(right_velocity) + '\n'
        print(command)
        self.ser.write(command.encode())

ser = serial.Serial('/dev/ttyACM0', 115200, timeout=1)

wheel_motors = Motor('setVelocity', ser)

ser.write("setVelocity 0 0\n".encode())
time.sleep(2)

wheel_motors.setVelocity(40, 40)
time.sleep(5)
wheel_motors.setVelocity(40, 0)
time.sleep(5)
wheel_motors.setVelocity(0, 40)
time.sleep(5)
wheel_motors.setVelocity(40, 40)
time.sleep(5)
wheel_motors.setVelocity(0, 0)