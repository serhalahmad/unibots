import serial

class Motor:
    def __init__(self, name, ser):
        self.name = name
        self.ser = ser

    def setVelocity(self, velocity):
        print(f"Setting {self.name} motors velocity to {velocity}")
        command = str(self.name) + " " + str(velocity)
        self.ser.write(command.encode())

ser = serial.Serial('/dev/ttyACM0', 115200, timeout=1)

left_motor = Motor('left', ser)
right_motor = Motor('right', ser)

left_motor.setVelocity(75)
right_motor.setVelocity(-75)