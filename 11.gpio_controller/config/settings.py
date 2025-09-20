#!/usr/bin/env python3
"""
Configuration settings for GPIO controller system
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))

class GPIOConfig:
    """GPIO pin configuration"""

    # RGB LED Pins
    RGB_RED_PIN = int(os.getenv('RGB_RED_PIN', 27))
    RGB_GREEN_PIN = int(os.getenv('RGB_GREEN_PIN', 17))
    RGB_BLUE_PIN = int(os.getenv('RGB_BLUE_PIN', 22))

    # Vibration Motor Pins
    VIBRATION_MOTOR_1_PIN = int(os.getenv('VIBRATION_MOTOR_1_PIN', 23))
    VIBRATION_MOTOR_2_PIN = int(os.getenv('VIBRATION_MOTOR_2_PIN', 24))

    # Input Button Pins
    BUTTON_1_PIN = int(os.getenv('BUTTON_1_PIN', 5))
    BUTTON_2_PIN = int(os.getenv('BUTTON_2_PIN', 6))

    # GPIO Chip Names
    OUTPUT_CHIP = os.getenv('OUTPUT_CHIP', 'gpiochip0')
    INPUT_CHIP = os.getenv('INPUT_CHIP', 'gpiochip4')

class ZeroMQConfig:
    """ZeroMQ port configuration"""

    # Process command ports
    RGB_COMMAND_PORT = int(os.getenv('RGB_COMMAND_PORT', 5558))
    MOTOR_COMMAND_PORT = int(os.getenv('MOTOR_COMMAND_PORT', 5560))
    INPUT_COMMAND_PORT = int(os.getenv('INPUT_COMMAND_PORT', 5562))

    # Process event ports
    RGB_EVENT_PORT = int(os.getenv('RGB_EVENT_PORT', 5559))
    MOTOR_EVENT_PORT = int(os.getenv('MOTOR_EVENT_PORT', 5561))
    INPUT_EVENT_PORT = int(os.getenv('INPUT_EVENT_PORT', 5563))

    # Status collector port
    STATUS_COLLECTOR_PORT = int(os.getenv('STATUS_COLLECTOR_PORT', 5557))

class SystemConfig:
    """System-wide configuration"""

    # Debounce time for inputs (seconds)
    INPUT_DEBOUNCE_TIME = float(os.getenv('INPUT_DEBOUNCE_TIME', 0.05))

    # Default breathing effect duration
    BREATHING_DURATION = float(os.getenv('BREATHING_DURATION', 2.5))

    # Status update interval (seconds)
    STATUS_UPDATE_INTERVAL = int(os.getenv('STATUS_UPDATE_INTERVAL', 5))

    # Process startup delay (seconds)
    PROCESS_STARTUP_DELAY = int(os.getenv('PROCESS_STARTUP_DELAY', 1))

# Color definitions
DEFAULT_COLORS = {
    'green': (0.0, 1.0, 0.0),      # Default
    'red': (1.0, 0.0, 0.0),        # Button 1
    'blue': (0.0, 0.0, 1.0),       # Button 2
    'white': (1.0, 1.0, 1.0),      # Both buttons
    'purple': (0.8, 0.0, 1.0),     # Special mode 1
    'orange': (1.0, 0.5, 0.0),     # Special mode 2
    'yellow': (1.0, 1.0, 0.0)      # Warning
}