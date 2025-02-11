#include <Arduino.h>
#include <unibots_servo.hpp>
#include <unibots_dc_motor.hpp>

#define SERVO_MODE 0
#define DC_MOTOR_MODE 1

#define SERVO_PIN 9
#define DC_MOTOR_1_PWM_PIN 10
#define DC_MOTOR_1_CW_PIN 22
#define DC_MOTOR_1_CCW_PIN 23

UnibotsServo backServo(SERVO_PIN, 90);                                                // servo for the back of the rugby storage
UnibotsDCMotor dc_motor_1(DC_MOTOR_1_PWM_PIN, DC_MOTOR_1_CW_PIN, DC_MOTOR_1_CCW_PIN); // First motor

int test_mode = DC_MOTOR_MODE;

void setup()
{
  Serial.begin(9600);
  switch (test_mode)
  {
  case SERVO_MODE:
    backServo.attach();
    backServo.set_angle(90);
    break;
  case DC_MOTOR_MODE:
    dc_motor_1.begin();
    break;
  default:
    break;
  }
}

void loop()
{
  switch (test_mode)
  {
  case SERVO_MODE:
    Serial.println("Testing Servo...");
    backServo.set_angle(0);
    delay(1000);
    backServo.set_angle(180);
    delay(1000);
    break;
  case DC_MOTOR_MODE:
    Serial.println("Testing DC Motor...");
    dc_motor_1.set_speed(150);
    delay(2000);
    dc_motor_1.set_speed(-150);
    delay(2000);
    dc_motor_1.stop();
    delay(1000);
    break;
  default:
    Serial.println("Invalid mode selected!");
    break;
  }
}