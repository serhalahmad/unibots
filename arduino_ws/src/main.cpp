#include <Arduino.h>
#include <unibots_servo.hpp>

UnibotsServo backServo(9, 90); // servo for the back of the rugby storage

void setup()
{
  backServo.attach();
  backServo.set_angle(90);
}

void loop()
{
  backServo.set_angle(0);
  delay(1000);
  backServo.set_angle(180);
  delay(1000);
}