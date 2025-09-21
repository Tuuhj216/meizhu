#!/usr/bin/env python3
import time
import math
import threading
from .single_gpio_controller import SingleGPIOController

class RGBLightController:
    """RGB light controller using multiple single GPIO controllers"""

    def __init__(self, chip_name='gpiochip0', red_pin=27, green_pin=17, blue_pin=22):
        """
        Initialize RGB light controller

        Args:
            chip_name: GPIO chip name
            red_pin: Red LED pin number
            green_pin: Green LED pin number
            blue_pin: Blue LED pin number
        """
        self.red_controller = SingleGPIOController(chip_name, red_pin, 'out', 'rgb_red')
        self.green_controller = SingleGPIOController(chip_name, green_pin, 'out', 'rgb_green')
        self.blue_controller = SingleGPIOController(chip_name, blue_pin, 'out', 'rgb_blue')

        self.running = True
        self.current_thread = None

    def set_color(self, red=0, green=0, blue=0):
        """Set RGB color (0 or 1 for each channel)"""
        if hasattr(self, 'red_controller'):
            self.red_controller.set_value(red)
        if hasattr(self, 'green_controller'):
            self.green_controller.set_value(green)
        if hasattr(self, 'blue_controller'):
            self.blue_controller.set_value(blue)

    def turn_off(self):
        """Turn off all LEDs"""
        self.set_color(0, 0, 0)

    def power_off(self):
        """Turn off power supply and stop all effects"""
        self.stop_effect()
        self.turn_off()

    def software_pwm_rgb(self, red_duty, green_duty, blue_duty, frequency=1000):
        """
        Software PWM implementation for RGB
        duty_cycles: 0.0 to 1.0 (0% to 100%) for each color
        frequency: PWM frequency in Hz
        """
        if not self.running:
            return

        period = 1.0 / frequency

        # Calculate on times for each color
        red_on_time = period * red_duty
        green_on_time = period * green_duty
        blue_on_time = period * blue_duty

        # Find the maximum on time to determine cycle timing
        max_on_time = max(red_on_time, green_on_time, blue_on_time)

        # Turn on all pins that should be on
        if red_duty > 0 and hasattr(self, 'red_controller'):
            self.red_controller.set_value(1)
        if green_duty > 0 and hasattr(self, 'green_controller'):
            self.green_controller.set_value(1)
        if blue_duty > 0 and hasattr(self, 'blue_controller'):
            self.blue_controller.set_value(1)

        # Sleep for minimum on time (interruptible)
        if max_on_time > 0:
            self._interruptible_sleep(max_on_time)

        # Turn off pins as their duty cycle expires
        if red_duty < 1 and hasattr(self, 'red_controller'):
            self.red_controller.set_value(0)
        if green_duty < 1 and hasattr(self, 'green_controller'):
            self.green_controller.set_value(0)
        if blue_duty < 1 and hasattr(self, 'blue_controller'):
            self.blue_controller.set_value(0)

        # Sleep for the remaining period (interruptible)
        off_time = period - max_on_time
        if off_time > 0:
            self._interruptible_sleep(off_time)

    def _interruptible_sleep(self, duration):
        """Sleep that can be interrupted by setting self.running = False"""
        sleep_increment = 0.01  # Check every 10ms
        elapsed = 0
        while elapsed < duration and self.running:
            sleep_time = min(sleep_increment, duration - elapsed)
            time.sleep(sleep_time)
            elapsed += sleep_time

    def breathing_effect(self, duration=2.0, color=(1.0, 0.0, 0.0)):
        """
        Exponential breathing effect for RGB
        color: tuple of (red, green, blue) intensity ratios (0.0 to 1.0)
        """
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

    def start_breathing_effect(self, color=(0.0, 1.0, 0.0), duration=2.5):
        """Start breathing light effect in a separate thread"""
        self.stop_effect()

        self.running = True
        self.current_thread = threading.Thread(
            target=self.breathing_effect,
            args=(duration, color)
        )
        self.current_thread.daemon = True
        self.current_thread.start()

    def stop_effect(self):
        """Stop any running effect"""
        if hasattr(self, 'current_thread') and self.current_thread and self.current_thread.is_alive():
            self.running = False
            self.current_thread.join(timeout=1)

    def cleanup(self):
        """Clean up all GPIO resources"""
        self.stop_effect()
        # Explicitly set all RGB pins to low voltage (0)
        self.set_color(0, 0, 0)
        print("RGB light set to low voltage and effects stopped")

        if hasattr(self, 'red_controller'):
            self.red_controller.cleanup()
        if hasattr(self, 'green_controller'):
            self.green_controller.cleanup()
        if hasattr(self, 'blue_controller'):
            self.blue_controller.cleanup()

    def __del__(self):
        """Destructor to ensure cleanup"""
        self.cleanup()