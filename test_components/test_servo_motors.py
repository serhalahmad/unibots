import serial
import time

class Motor:
    def __init__(self, name, ser):
        self.name = name
        self.ser = ser
        # Initialize motor hardware here

    def setVelocity(self, velocity):
        # Send PWM signal or command to motor driver
        print(f"Setting {self.name} motors velocity to {velocity}")
        # Example: "setVelocity 40 75" (left motor 40, right motor 75)
        command = str(self.name) + " " + str(velocity)
        self.ser.write(command.encode())

ser = serial.Serial('/dev/ttyACM0', 115200, timeout=1)

wheel_motors = Motor('setVelocity', ser)

wheel_motors.setVelocity(40, 40)
time.sleep(1.5)
wheel_motors.setVelocity(40, 0)
time.sleep(1.5)
wheel_motors.setVelocity(0, 40)
time.sleep(1.5)
wheel_motors.setVelocity(40, 40)
time.sleep(3)
wheel_motors.setVelocity(0, 0)