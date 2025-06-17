import serial
import time

class Motor:
    def __init__(self, name, ser):
        self.name = name
        self.ser = ser
        # Initialize motor hardware here

    def setVelocity(self, left_velocity, right_velocity):
        # Send PWM signal or command to motor driver
        print(f"{self.name}: setting left wheels to {left_velocity} and right wheels to {right_velocity}")
        command = f"{self.name} {left_velocity} {right_velocity}\n"
        print(f"Command sent: {command.strip()}")
        self.ser.write(command.encode())

# Initialize serial connection
ser = serial.Serial('/dev/ttyACM0', 115200, timeout=1)
wheel_motors = Motor('setVelocity', ser)

# Example initial command
ser.write("setVelocity 0 0\n".encode())
# ser.write("setVelocity 0 0\n".encode())
# time.sleep(2)
# ser.write("setVelocity 75 75\n".encode())

# Continuous loop to read input and send commands
while True:
    try:
        user_input = input("Enter left and right motor speeds separated by a space (e.g. 40 -40): ")
        # Split the input and convert each value to an integer
        left_velocity, right_velocity = map(int, user_input.split())
        wheel_motors.setVelocity(left_velocity, right_velocity)
    except ValueError:
        print("Invalid input. Please enter two numbers separated by a space.")
