#!/usr/bin/env python3
import gpiod

class SingleGPIOController:
    """Simple GPIO controller for controlling a single pin"""

    def __init__(self, chip_name, pin_number, direction='out', consumer="gpio_controller"):
        """
        Initialize a single GPIO pin controller

        Args:
            chip_name: GPIO chip name (e.g., 'gpiochip0')
            pin_number: Pin number to control
            direction: 'out' for output, 'in' for input
            consumer: Consumer name for GPIO line
        """
        self.chip_name = chip_name
        self.pin_number = pin_number
        self.direction = direction
        self.consumer = consumer

        self.chip = None
        self.line = None
        self._initialize()

    def _initialize(self):
        """Initialize the GPIO chip and line"""
        try:
            self.chip = gpiod.Chip(self.chip_name)
            self.line = self.chip.get_line(self.pin_number)

            if self.direction == 'out':
                self.line.request(consumer=self.consumer, type=gpiod.LINE_REQ_DIR_OUT)
            else:
                self.line.request(consumer=self.consumer, type=gpiod.LINE_REQ_DIR_IN)

        except Exception as e:
            raise RuntimeError(f"Failed to initialize GPIO pin {self.pin_number}: {e}")

    def set_value(self, value):
        """Set pin value (0 or 1) for output pins"""
        if self.direction != 'out':
            raise ValueError("Cannot set value on input pin")

        if self.line:
            self.line.set_value(int(value))

    def get_value(self):
        """Get pin value for input pins"""
        if self.line:
            return self.line.get_value()
        return 0

    def cleanup(self):
        """Clean up GPIO resources"""
        if self.line:
            if self.direction == 'out':
                self.line.set_value(0)  # Turn off output
            self.line.release()
            self.line = None

        if self.chip:
            self.chip.close()
            self.chip = None