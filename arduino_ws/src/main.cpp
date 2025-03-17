#include <Arduino.h>
#include <unibots_servo.hpp>
#include <unibots_dc_motor.hpp>

#define SERVO_MODE 0
#define DC_MOTOR_MODE 1
#define SERIAL_COM_MODE 2

#define SERVO_PIN 9
#define DC_MOTOR_1_PWM_PIN 9
#define DC_MOTOR_2_PWM_PIN 10
#define DC_MOTOR_3_PWM_PIN 11
#define DC_MOTOR_1_CW_PIN 12
#define DC_MOTOR_2_CW_PIN 12
#define DC_MOTOR_3_CW_PIN 13
#define DC_MOTOR_1_CCW_PIN 12
#define DC_MOTOR_2_CCW_PIN 12
#define DC_MOTOR_3_CCW_PIN 13

UnibotsServo backServo(SERVO_PIN, 90);                                                // servo for the back of the rugby storage
UnibotsDCMotor dc_motor_1(DC_MOTOR_1_PWM_PIN, DC_MOTOR_1_CW_PIN, DC_MOTOR_1_CCW_PIN); // First motor
UnibotsDCMotor dc_motor_2(DC_MOTOR_2_PWM_PIN, DC_MOTOR_2_CW_PIN, DC_MOTOR_2_CCW_PIN); // Second motor
UnibotsDCMotor dc_motor_3(DC_MOTOR_3_PWM_PIN, DC_MOTOR_3_CW_PIN, DC_MOTOR_3_CCW_PIN); // Third motor

int test_mode = SERIAL_COM_MODE;

void setup()
{

  switch (test_mode)
  {
  case SERVO_MODE:
    Serial.begin(9600);
    backServo.attach();
    backServo.set_angle(90);
    break;
  case DC_MOTOR_MODE:
    Serial.begin(9600);
    dc_motor_1.begin();
    dc_motor_2.begin();
    dc_motor_3.begin();
    break;
  case SERIAL_COM_MODE:
    Serial.begin(115200);
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
    dc_motor_2.set_speed(150);
    dc_motor_3.set_speed(150);
    delay(2000);
    dc_motor_1.set_speed(-150);
    dc_motor_2.set_speed(-150);
    dc_motor_3.set_speed(-150);
    delay(2000);
    dc_motor_1.stop();
    dc_motor_2.stop();
    dc_motor_3.stop();
    delay(1000);
    break;
  case SERIAL_COM_MODE:
    if (Serial.available())
    {
      String received = Serial.readStringUntil('\n');
      Serial.print("Arduino recieved: ");
      Serial.print(received);
      Serial.println(". Got it :)");
    }
    Serial.println("Hello from Arduino!");
    delay(1000);
    break;
  default:
    Serial.println("Invalid mode selected!");
    break;
  }
}