#!/usr/bin/env python3
import gpiod
import time
import math
import signal
import sys

class RGBBreathingLight:
    def __init__(self, chip_name='gpiochip0', red_pin=23, green_pin=24, blue_pin=25):
        self.chip = gpiod.Chip(chip_name)
        self.red_line = self.chip.get_line(red_pin)
        self.green_line = self.chip.get_line(green_pin)
        self.blue_line = self.chip.get_line(blue_pin)

        self.red_line.request(consumer="rgb_breathing_light", type=gpiod.LINE_REQ_DIR_OUT)
        self.green_line.request(consumer="rgb_breathing_light", type=gpiod.LINE_REQ_DIR_OUT)
        self.blue_line.request(consumer="rgb_breathing_light", type=gpiod.LINE_REQ_DIR_OUT)

        self.running = True

        # Setup signal handler for clean exit
        signal.signal(signal.SIGINT, self.signal_handler)

    def signal_handler(self, sig, frame):
        print("\nExiting gracefully...")
        self.running = False

    def software_pwm_rgb(self, red_duty, green_duty, blue_duty, frequency=1000):
        """
        Software PWM implementation for RGB
        duty_cycles: 0.0 to 1.0 (0% to 100%) for each color
        frequency: PWM frequency in Hz
        """
        period = 1.0 / frequency

        # Calculate on times for each color
        red_on_time = period * red_duty
        green_on_time = period * green_duty
        blue_on_time = period * blue_duty

        # Find the maximum on time to determine cycle timing
        max_on_time = max(red_on_time, green_on_time, blue_on_time)

        # Turn on all pins that should be on
        if red_duty > 0:
            self.red_line.set_value(1)
        if green_duty > 0:
            self.green_line.set_value(1)
        if blue_duty > 0:
            self.blue_line.set_value(1)

        # Sleep for minimum on time
        if max_on_time > 0:
            time.sleep(max_on_time)

        # Turn off pins as their duty cycle expires
        if red_duty < 1:
            self.red_line.set_value(0)
        if green_duty < 1:
            self.green_line.set_value(0)
        if blue_duty < 1:
            self.blue_line.set_value(0)

        # Sleep for the remaining period
        off_time = period - max_on_time
        if off_time > 0:
            time.sleep(off_time)

    def breathing_effect_exponential(self, duration=2.0, color=(1.0, 0.5, 0.2)):
        """
        Exponential breathing effect for RGB (more natural looking)
        color: tuple of (red, green, blue) intensity ratios (0.0 to 1.0)
        """
        print(f"Starting RGB exponential breathing effect with color {color} (Ctrl+C to stop)")

        red_ratio, green_ratio, blue_ratio = color

        while self.running:
            start_time = time.time()

            while time.time() - start_time < duration:
                if not self.running:
                    break

                # Calculate position in cycle (0 to 1)
                progress = (time.time() - start_time) / duration

                # Create exponential breathing curve
                if progress < 0.5:
                    # Breathing in - exponential rise
                    brightness = math.pow(progress * 2, 2)
                else:
                    # Breathing out - exponential fall
                    brightness = math.pow((1 - progress) * 2, 2)

                # Apply brightness to each color channel
                red_duty = brightness * red_ratio
                green_duty = brightness * green_ratio
                blue_duty = brightness * blue_ratio

                self.software_pwm_rgb(red_duty, green_duty, blue_duty, frequency=200)

    def cleanup(self):
        """Clean up GPIO resources"""
        self.red_line.set_value(0)
        self.green_line.set_value(0)
        self.blue_line.set_value(0)

        self.red_line.release()
        self.green_line.release()
        self.blue_line.release()
        self.chip.close()
        print("RGB GPIO cleaned up")

if __name__ == "__main__":
    try:
        # Create RGB breathing light with default pins (23=red, 24=green, 25=blue)
        rgb_light = RGBBreathingLight()

        # Start breathing effect with warm white color
        rgb_light.breathing_effect_exponential(duration=3.0, color=(1.0, 0.8, 0.6))

    except KeyboardInterrupt:
        pass
    finally:
        if 'rgb_light' in locals():
            rgb_light.cleanup()
