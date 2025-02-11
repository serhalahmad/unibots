#pragma once

#include <Arduino.h>
#include <Servo.h>

class UnibotsServo
{
public:
    UnibotsServo(int pin, float initial_angle); // Constructor
    void attach();                              // Attach servo to pin
    void detach();                              // Detach servo to save power
    void set_angle(int angle);                  // Set servo angle (0 - 180 degrees)
    int get_angle() const;                      // Get current angle

private:
    int _pwm_pin;
    int _current_angle;
    Servo _servo;
};