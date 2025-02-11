#include "unibots_dc_motor.hpp"

UnibotsDCMotor::UnibotsDCMotor(int pwm_pin, int cw_pin, int ccw_pin) : _pwm_pin(pwm_pin), _cw_pin(cw_pin), _ccw_pin(ccw_pin), _speed(0) {}

void UnibotsDCMotor::begin()
{
    pinMode(_pwm_pin, OUTPUT);
    pinMode(_cw_pin, OUTPUT);
    pinMode(_ccw_pin, OUTPUT);
    stop(); // Ensure motor starts stopped
}

void UnibotsDCMotor::set_speed(int speed)
{
    _speed = constrain(speed, -255, 255); // Ensure valid range

    if (_speed > 0)
    {
        digitalWrite(_cw_pin, HIGH);
        digitalWrite(_ccw_pin, LOW);
    }
    else if (_speed < 0)
    {
        digitalWrite(_cw_pin, LOW);
        digitalWrite(_ccw_pin, HIGH);
    }
    else
    {
        stop(); // Stop if speed = 0
        return;
    }
    analogWrite(_pwm_pin, abs(_speed)); // Set PWM speed
}

void UnibotsDCMotor::stop()
{
    digitalWrite(_cw_pin, LOW);
    digitalWrite(_ccw_pin, LOW);
    analogWrite(_pwm_pin, 0);
}
