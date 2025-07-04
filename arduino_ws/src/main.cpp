#include <Arduino.h>
#include <unibots_servo.hpp>
#include <unibots_dc_motor.hpp>

#define SERVO_MODE 0
#define DC_MOTOR_MODE 1
#define SERIAL_COM_MODE 2

// All motor names are when looking at the top of the robot and it is upward
// Also, the control pins are in a way that they are consecutively alligned with the motor pins
#define SERVO_PIN 6
#define DC_MOTOR_FR_PWM_PIN 10
#define DC_MOTOR_FL_PWM_PIN 9
#define DC_MOTOR_BR_PWM_PIN 10
#define DC_MOTOR_BL_PWM_PIN 9

#define DC_MOTOR_FR_CW_PIN 13
#define DC_MOTOR_FL_CW_PIN 12
#define DC_MOTOR_BR_CW_PIN 13
#define DC_MOTOR_BL_CW_PIN 12

#define DC_MOTOR_FR_CCW_PIN 8
#define DC_MOTOR_FL_CCW_PIN 7
#define DC_MOTOR_BR_CCW_PIN 8
#define DC_MOTOR_BL_CCW_PIN 7

UnibotsServo backServo(SERVO_PIN, 90);                                                    // servo for the back of the rugby storage
UnibotsDCMotor dc_motor_fr(DC_MOTOR_FR_PWM_PIN, DC_MOTOR_FR_CW_PIN, DC_MOTOR_FR_CCW_PIN); // Front-right motor
UnibotsDCMotor dc_motor_fl(DC_MOTOR_FL_PWM_PIN, DC_MOTOR_FL_CW_PIN, DC_MOTOR_FL_CCW_PIN); // Front-left motor
UnibotsDCMotor dc_motor_br(DC_MOTOR_BR_PWM_PIN, DC_MOTOR_BR_CW_PIN, DC_MOTOR_BR_CCW_PIN); // Back-right motor
UnibotsDCMotor dc_motor_bl(DC_MOTOR_BL_PWM_PIN, DC_MOTOR_BL_CW_PIN, DC_MOTOR_BL_CCW_PIN); // Back-left motor

int test_mode = SERIAL_COM_MODE;

void setup()
{
  Serial.begin(115200);
  switch (test_mode)
  {
  case SERVO_MODE:
    // Serial.begin(9600);
    backServo.attach();
    backServo.set_angle(90);
    break;
  case DC_MOTOR_MODE:
    // Serial.begin(115200);
    dc_motor_fr.begin();
    dc_motor_fl.begin();
    dc_motor_br.begin();
    dc_motor_bl.begin();
    break;
  case SERIAL_COM_MODE:
    // Serial.begin(115200);
    dc_motor_fr.begin();
    dc_motor_fl.begin();
    dc_motor_br.begin();
    dc_motor_bl.begin();
    break;
  default:
    break;
  }
}

void loop()
{
  // dc_motor_br.set_speed(75);
  // TODO: Back and front are mixed up
  switch (test_mode)
  {
  case SERVO_MODE:
    Serial.println("Testing Servo...");
    // backServo.set_angle(270); // catch
    // delay(1000);
    backServo.set_angle(-90); // release
    delay(1000);
    // backServo.set_angle(0);
    // delay(1000);
    break;
  case DC_MOTOR_MODE:
    // String received = "setVelocity 40 75";
    // int firstSpace = received.indexOf(' ');                                   // Find first space
    // int secondSpace = received.indexOf(' ', firstSpace + 1);                  // Find first space
    // int left_speed = received.substring(firstSpace + 1, secondSpace).toInt(); // Extract left speed
    // int right_speed = received.substring(secondSpace + 1).toInt();            // Extract right speed
    // dc_motor_fl.set_speed(left_speed);
    // dc_motor_bl.set_speed(left_speed);
    // dc_motor_fr.set_speed(right_speed);
    // dc_motor_br.set_speed(right_speed);
    //

    // NOTE: For some reason, we have to comment this out to test serial mode
    // int right_speed = -100;
    // int left_speed = 100;
    // dc_motor_fr.set_speed(right_speed);
    // dc_motor_fl.set_speed(left_speed);
    // dc_motor_br.set_speed(right_speed);
    // dc_motor_bl.set_speed(left_speed);
    // delay(2000);
    // dc_motor_fr.stop();
    // dc_motor_fl.stop();
    // dc_motor_br.stop();
    // dc_motor_bl.stop();
    // delay(1000);
    // break;
  case SERIAL_COM_MODE:
    if (Serial.available())
    {
      String received = Serial.readStringUntil('\n');
      if (received.startsWith("setVelocity"))
      {
        int firstSpace = received.indexOf(' ');                                   // Find first space
        int secondSpace = received.indexOf(' ', firstSpace + 1);                  // Find first space
        int left_speed = received.substring(firstSpace + 1, secondSpace).toInt(); // Extract left speed
        int right_speed = received.substring(secondSpace + 1).toInt();            // Extract right speed
        dc_motor_fl.set_speed(left_speed);
        dc_motor_bl.set_speed(left_speed);
        dc_motor_fr.set_speed(right_speed);
        dc_motor_br.set_speed(right_speed);
      }
      else if (received.startsWith("stop"))
      {
        dc_motor_fr.stop();
        dc_motor_fl.stop();
        dc_motor_br.stop();
        dc_motor_bl.stop();
      }
    }
    break;
  default:
    break;
  }
}