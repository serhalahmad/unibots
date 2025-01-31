#include "unibots_servo.hpp"

UnibotsServo::UnibotsServo(int pin, float initial_angle) : servo_pin(pin), current_angle(initial_angle) {}

void UnibotsServo::attach()
{
    servo.attach(servo_pin);
}

void UnibotsServo::detach()
{
    servo.detach();
}

void UnibotsServo::set_angle(int angle)
{
    angle = constrain(angle, 0, 180); // ensure valid range
    servo.write(angle);
    current_angle = angle;
}

int UnibotsServo::get_angle() const
{
    return current_angle;
}