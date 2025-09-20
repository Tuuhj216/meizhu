#!/usr/bin/env python3
import os
from .base_process import BaseProcess
from ..gpio.rgb_light_controller import RGBLightController

class RGBLightProcess(BaseProcess):
    """RGB Light control process"""

    def __init__(self):
        super().__init__(
            process_name="rgb_light",
            command_port=int(os.getenv('RGB_COMMAND_PORT', 5558)),
            event_port=int(os.getenv('RGB_EVENT_PORT', 5559)),
        )
        self.rgb_controller = None
        self.current_effect = None

    def initialize_hardware(self):
        """Initialize RGB light controller"""
        try:
            # Load pin configuration from environment
            red_pin = int(os.getenv('RGB_RED_PIN', 27))
            green_pin = int(os.getenv('RGB_GREEN_PIN', 17))
            blue_pin = int(os.getenv('RGB_BLUE_PIN', 22))
            chip_name = os.getenv('OUTPUT_CHIP', 'gpiochip0')

            self.rgb_controller = RGBLightController(
                chip_name=chip_name,
                red_pin=red_pin,
                green_pin=green_pin,
                blue_pin=blue_pin
            )

            print(f"{self.process_name}: RGB controller initialized")
            self.send_event("rgb_initialized", {
                "red_pin": red_pin,
                "green_pin": green_pin,
                "blue_pin": blue_pin,
                "chip": chip_name
            })

        except Exception as e:
            print(f"{self.process_name}: Error initializing RGB hardware: {e}")
            raise

    def cleanup_hardware(self):
        """Clean up RGB controller"""
        if self.rgb_controller:
            self.rgb_controller.cleanup()
            print(f"{self.process_name}: RGB controller cleaned up")

    def handle_process_command(self, command_data):
        """Handle RGB-specific commands"""
        command = command_data.get("command")
        parameters = command_data.get("parameters", {})
        correlation_id = command_data.get("correlation_id")

        try:
            if command == "set_color":
                # Set static color
                red = parameters.get("red", 0)
                green = parameters.get("green", 0)
                blue = parameters.get("blue", 0)

                if self.rgb_controller:
                    self.rgb_controller.stop_effect()  # Stop any running effects
                    self.rgb_controller.set_color(red, green, blue)
                    self.current_effect = None

                    self.send_event("color_changed", {
                        "red": red,
                        "green": green,
                        "blue": blue
                    })

                return {
                    "status": "ok",
                    "action": "color_set",
                    "color": {"red": red, "green": green, "blue": blue},
                    "correlation_id": correlation_id
                }

            elif command == "start_breathing":
                # Start breathing effect
                color = parameters.get("color", [0.0, 1.0, 0.0])  # Default green
                duration = parameters.get("duration", 2.5)

                if self.rgb_controller:
                    self.rgb_controller.start_breathing_effect(tuple(color), duration)
                    self.current_effect = "breathing"

                    self.send_event("breathing_started", {
                        "color": color,
                        "duration": duration
                    })

                return {
                    "status": "ok",
                    "action": "breathing_started",
                    "color": color,
                    "duration": duration,
                    "correlation_id": correlation_id
                }

            elif command == "stop_effect":
                # Stop any running effect
                if self.rgb_controller:
                    self.rgb_controller.stop_effect()
                    self.current_effect = None

                    self.send_event("effect_stopped")

                return {
                    "status": "ok",
                    "action": "effect_stopped",
                    "correlation_id": correlation_id
                }

            elif command == "turn_off":
                # Turn off all LEDs
                if self.rgb_controller:
                    self.rgb_controller.stop_effect()
                    self.rgb_controller.turn_off()
                    self.current_effect = None

                    self.send_event("lights_off")

                return {
                    "status": "ok",
                    "action": "lights_off",
                    "correlation_id": correlation_id
                }

            elif command == "get_status":
                # Return current status
                return {
                    "status": "ok",
                    "current_effect": self.current_effect,
                    "correlation_id": correlation_id
                }

            else:
                return {
                    "status": "error",
                    "error": f"Unknown command: {command}",
                    "correlation_id": correlation_id
                }

        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "correlation_id": correlation_id
            }


if __name__ == "__main__":
    process = RGBLightProcess()
    try:
        process.run()
    except KeyboardInterrupt:
        print("\nRGB Light Process: Received interrupt")
    finally:
        process.cleanup()