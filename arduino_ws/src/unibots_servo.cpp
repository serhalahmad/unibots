#include "unibots_servo.hpp"

UnibotsServo::UnibotsServo(int pin, float initial_angle) : _pwm_pin(pin), _current_angle(initial_angle) {}

void UnibotsServo::attach()
{
    _servo.attach(_pwm_pin);
}

void UnibotsServo::detach()
{
    _servo.detach();
}

void UnibotsServo::set_angle(int angle)
{
    angle = constrain(angle, -360, 360); // ensure valid range
    _servo.write(angle);
    _current_angle = angle;
}

int UnibotsServo::get_angle() const
{
    return _current_angle;
}