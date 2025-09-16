#!/bin/bash

PIN=23

echo $PIN > /sys/class/gpio/export
echo in > /sys/class/gpio/gpio$PIN/direction

# Read the pin state
state=$(cat /sys/class/gpio/gpio$PIN/value)
echo "Pin 23 is: $state"

echo $PIN > /sys/class/gpio/unexport
