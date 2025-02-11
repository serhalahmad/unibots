#pragma once

#include <Arduino.h>

class UnibotsDCMotor
{
public:
    UnibotsDCMotor(int pwm_pin, int cw_pin, int ccw_pin); // Constructor
    void begin();                                         // Initialize motor pins
    void set_speed(int speed);                            // Set speed (-255 to 255)
    void stop();                                          // Stop motor

private:
    int _pwm_pin; // PWM pin for speed control
    int _cw_pin;  // Clockwise Direction
    int _ccw_pin; // Counter-Clockwise Direction
    int _speed;   // Store current speed
};