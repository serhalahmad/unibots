# DC Motor Connection
- Pin IP1 of L298N motor driver to digital pin 22 of Arduino Mega or digital pin 13 of Uno.
- Pin IP2 of L298N motor driver to digital pin 23 of Arduino Mega or digital pin 12 of Uno.
- Pin ENA of L298N motor driver to PWM pin 10 of Arduino Mega or digital pin 11 of Uno.
- Connect the motor to port J2 of the L298N motor driver. My convention is positive side closer to the heat sink.
- Connect the battery to J1 port of the L298N motor driver.
- TODO: NOTE that logically, I should have a common ground between arduino, motor driver, and the battery. My test worked without this. But for reliability, consider having a common ground and test it. 

# Servo Motor Connection
- Brown wire of the servo to a ground pin of Arduino Mega.
- Red wire of the servo to a 5V pin of Arduino Mega.
- Orange wire of the servo to PWM pin 9 of Arduino Mega.