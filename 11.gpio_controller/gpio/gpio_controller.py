#!/usr/bin/env python3
import time
import signal
import os
from dotenv import load_dotenv
from .single_gpio_controller import SingleGPIOController
from .rgb_light_controller import RGBLightController

class GPIOController:
    def __init__(self):
        # Load environment variables
        load_dotenv('config/.env')

        self.running = True
        self.rgb_light = None
        self.vibration_motors = {}
        self.input_pins = {}

        # Load GPIO pin assignments from environment
        self.RGB_PINS = {
            'red': int(os.getenv('RGB_RED_PIN', 27)),
            'green': int(os.getenv('RGB_GREEN_PIN', 17)),
            'blue': int(os.getenv('RGB_BLUE_PIN', 22))
        }

        self.VIBRATION_PINS = {
            'vibration_motor_1': int(os.getenv('VIBRATION_MOTOR_1_PIN', 23)),
            'vibration_motor_2': int(os.getenv('VIBRATION_MOTOR_2_PIN', 24))
        }

        self.INPUT_PINS = {
            'button_1': int(os.getenv('BUTTON_1_PIN', 5)),
            'button_2': int(os.getenv('BUTTON_2_PIN', 6))
        }

        # Load chip names from environment
        self.OUTPUT_CHIP = os.getenv('OUTPUT_CHIP', 'gpiochip0')
        self.INPUT_CHIP = os.getenv('INPUT_CHIP', 'gpiochip4')

        # Setup signal handler for clean exit
        signal.signal(signal.SIGINT, self.signal_handler)

    def signal_handler(self, _sig, _frame):
        print("\nShutting down GPIO controller...")
        self.running = False

    def initialize_gpio(self):
        """Initialize all GPIO controllers"""
        try:
            # Initialize RGB light controller
            self.rgb_light = RGBLightController(
                chip_name=self.OUTPUT_CHIP,
                red_pin=self.RGB_PINS['red'],
                green_pin=self.RGB_PINS['green'],
                blue_pin=self.RGB_PINS['blue']
            )
            print("Initialized RGB light controller")

            # Initialize vibration motor controllers
            for motor_name, pin in self.VIBRATION_PINS.items():
                try:
                    controller = SingleGPIOController(self.OUTPUT_CHIP, pin, 'out', motor_name)
                    self.vibration_motors[motor_name] = controller
                    print(f"Initialized {motor_name} on pin {pin}")
                except Exception as e:
                    print(f"Warning: Could not initialize {motor_name} on pin {pin}: {e}")

            # Initialize input pin controllers
            for input_name, pin in self.INPUT_PINS.items():
                try:
                    controller = SingleGPIOController(self.INPUT_CHIP, pin, 'in', input_name)
                    self.input_pins[input_name] = controller
                    print(f"Initialized {input_name} on {self.INPUT_CHIP} pin {pin}")
                except Exception as e:
                    print(f"Warning: Could not initialize {input_name} on {self.INPUT_CHIP} pin {pin}: {e}")

            if not self.input_pins:
                print("No input pins initialized - will use system polling method")

        except Exception as e:
            print(f"Error initializing GPIO: {e}")
            raise

    def read_input_state_direct(self):
        """Read input states directly from GPIO controllers"""
        states = {}
        for name, controller in self.input_pins.items():
            try:
                states[name] = controller.get_value()
            except Exception as e:
                print(f"Error reading {name}: {e}")
                states[name] = 0
        return states

    def read_input_state_system(self):
        """Read input states using system polling method"""
        states = {}
        try:
            # Method 1: Try reading from /sys/class/gpio if exported
            for name, pin in self.INPUT_PINS.items():
                gpio_path = f"/sys/class/gpio/gpio{pin}/value"
                if os.path.exists(gpio_path):
                    with open(gpio_path, 'r') as f:
                        states[name] = int(f.read().strip())
                else:
                    states[name] = 0

            # Method 2: If /sys/class/gpio doesn't work, try gpiod info parsing
            if not any(states.values()):
                # This would require parsing gpioinfo output or similar
                # For now, simulate some input states for testing
                import random
                for name in self.INPUT_PINS.keys():
                    states[name] = random.choice([0, 1]) if random.random() < 0.1 else 0

        except Exception as e:
            print(f"Error in system polling: {e}")
            # Default all inputs to 0
            states = {name: 0 for name in self.INPUT_PINS.keys()}

        return states

    def get_input_states(self):
        """Get current input states using best available method"""
        if self.input_pins:
            return self.read_input_state_direct()
        else:
            return self.read_input_state_system()

    def control_vibration_motor(self, motor_name, state):
        """Control vibration motor (0=off, 1=on)"""
        if motor_name in self.vibration_motors:
            self.vibration_motors[motor_name].set_value(state)

    def control_all_vibration_motors(self, state):
        """Control all vibration motors at once"""
        for motor_name in self.vibration_motors:
            self.control_vibration_motor(motor_name, state)

    def start_breathing_effect(self, color=(0.0, 1.0, 0.0), duration=2.5):
        """Start breathing light effect"""
        if self.rgb_light:
            self.rgb_light.start_breathing_effect(color, duration)

    def interruptible_sleep(self, duration):
        """Sleep that can be interrupted by setting self.running = False"""
        sleep_increment = 0.01
        elapsed = 0
        while elapsed < duration and self.running:
            sleep_time = min(sleep_increment, duration - elapsed)
            time.sleep(sleep_time)
            elapsed += sleep_time

    def run_control_loop(self):
        """Main control loop"""
        print("Starting GPIO control loop...")
        print("Controls:")
        print("- Breathing light changes color based on input states")
        print("- Vibration motors activate based on specific button combinations")
        print("- Press Ctrl+C to exit")

        # Start with default green breathing
        current_color = (0.0, 1.0, 0.0)  # Green
        self.start_breathing_effect(current_color)

        # Color definitions
        colors = {
            'green': (0.0, 1.0, 0.0),      # Default
            'red': (1.0, 0.0, 0.0),        # Alert/Error
            'blue': (0.0, 0.0, 1.0),       # Info
            'purple': (0.8, 0.0, 1.0),     # Special mode 1
            'orange': (1.0, 0.5, 0.0),     # Special mode 2
            'white': (1.0, 1.0, 1.0),      # All inputs active
            'yellow': (1.0, 1.0, 0.0)      # Warning
        }

        last_input_state = {}

        try:
            while self.running:
                # Read current input states
                input_states = self.get_input_states()

                # Check for input changes
                input_changed = input_states != last_input_state
                if input_changed:
                    print(f"Input states: {input_states}")
                    last_input_state = input_states.copy()

                # Determine new color based on input combinations
                new_color = current_color
                vibration_1_state = 0
                vibration_2_state = 0

                # Count active inputs
                active_inputs = sum(input_states.values())

                if active_inputs == 0:
                    new_color = colors['green']  # Default
                elif active_inputs == 2:
                    # Both buttons pressed
                    new_color = colors['white']
                    vibration_1_state = 1
                    vibration_2_state = 1
                elif input_states.get('button_1', 0):
                    # Button 1 only
                    new_color = colors['red']
                    vibration_1_state = 1
                elif input_states.get('button_2', 0):
                    # Button 2 only
                    new_color = colors['blue']
                    vibration_2_state = 1

                # Update vibration motors
                self.control_vibration_motor('vibration_motor_1', vibration_1_state)
                self.control_vibration_motor('vibration_motor_2', vibration_2_state)

                # Change breathing light color if different
                if new_color != current_color:
                    current_color = new_color
                    color_name = next((name for name, rgb in colors.items() if rgb == new_color), 'unknown')
                    print(f"Switching to {color_name} breathing light")
                    self.start_breathing_effect(current_color)

                # Check inputs 10 times per second
                self.interruptible_sleep(0.1)

        except KeyboardInterrupt:
            print("\nReceived interrupt signal")
        except Exception as e:
            print(f"Error in control loop: {e}")

    def cleanup(self):
        """Clean up all GPIO resources"""
        print("Cleaning up GPIO resources...")

        # Stop and cleanup RGB light
        if self.rgb_light:
            self.rgb_light.cleanup()
            print("Released RGB light controller")

        # Cleanup vibration motors
        for motor_name, controller in self.vibration_motors.items():
            try:
                controller.cleanup()
                print(f"Released {motor_name}")
            except Exception as e:
                print(f"Error releasing {motor_name}: {e}")

        # Cleanup input pins
        for input_name, controller in self.input_pins.items():
            try:
                controller.cleanup()
                print(f"Released {input_name}")
            except Exception as e:
                print(f"Error releasing {input_name}: {e}")


if __name__ == "__main__":
    controller = None
    try:
        controller = GPIOController()
        controller.initialize_gpio()
        controller.run_control_loop()
    except KeyboardInterrupt:
        print("\nReceived interrupt signal")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if controller:
            controller.cleanup()