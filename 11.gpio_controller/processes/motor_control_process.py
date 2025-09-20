#!/usr/bin/env python3
import os
import time
import threading
from .base_process import BaseProcess
from ..gpio.single_gpio_controller import SingleGPIOController

class MotorControlProcess(BaseProcess):
    """Vibration motor control process"""

    def __init__(self):
        super().__init__(
            process_name="motor_control",
            command_port=int(os.getenv('MOTOR_COMMAND_PORT', 5560)),
            event_port=int(os.getenv('MOTOR_EVENT_PORT', 5561)),
        )
        self.motors = {}
        self.pattern_thread = None
        self.pattern_running = False

    def initialize_hardware(self):
        """Initialize vibration motor controllers"""
        try:
            chip_name = os.getenv('OUTPUT_CHIP', 'gpiochip0')

            # Load motor pin configuration from environment
            motor_pins = {
                'motor_1': int(os.getenv('VIBRATION_MOTOR_1_PIN', 23)),
                'motor_2': int(os.getenv('VIBRATION_MOTOR_2_PIN', 24))
            }

            # Initialize each motor controller
            for motor_name, pin in motor_pins.items():
                try:
                    controller = SingleGPIOController(chip_name, pin, 'out', f'vibration_{motor_name}')
                    self.motors[motor_name] = controller
                    print(f"{self.process_name}: Initialized {motor_name} on pin {pin}")
                except Exception as e:
                    print(f"{self.process_name}: Warning - Could not initialize {motor_name}: {e}")

            self.send_event("motors_initialized", {
                "motors": list(self.motors.keys()),
                "chip": chip_name
            })

        except Exception as e:
            print(f"{self.process_name}: Error initializing motor hardware: {e}")
            raise

    def cleanup_hardware(self):
        """Clean up motor controllers"""
        self.stop_pattern()
        for motor_name, controller in self.motors.items():
            try:
                controller.cleanup()
                print(f"{self.process_name}: Cleaned up {motor_name}")
            except Exception as e:
                print(f"{self.process_name}: Error cleaning up {motor_name}: {e}")

    def set_motor_state(self, motor_name, state):
        """Set individual motor state"""
        if motor_name in self.motors:
            self.motors[motor_name].set_value(state)
            return True
        return False

    def set_all_motors(self, state):
        """Set all motors to the same state"""
        for motor_name in self.motors:
            self.set_motor_state(motor_name, state)

    def _pattern_worker(self, pattern_data):
        """Worker thread for executing vibration patterns"""
        pattern = pattern_data.get("pattern", [])
        repeat = pattern_data.get("repeat", False)

        try:
            while self.pattern_running:
                for step in pattern:
                    if not self.pattern_running:
                        break

                    motor = step.get("motor", "all")
                    state = step.get("state", 0)
                    duration = step.get("duration", 0.1)

                    # Apply motor state
                    if motor == "all":
                        self.set_all_motors(state)
                    elif motor in self.motors:
                        self.set_motor_state(motor, state)

                    # Send pattern step event
                    self.send_event("pattern_step", {
                        "motor": motor,
                        "state": state,
                        "duration": duration
                    })

                    # Wait for duration (interruptible)
                    sleep_time = 0
                    while sleep_time < duration and self.pattern_running:
                        time.sleep(min(0.1, duration - sleep_time))
                        sleep_time += 0.1

                if not repeat:
                    break

        except Exception as e:
            print(f"{self.process_name}: Error in pattern execution: {e}")
        finally:
            # Turn off all motors when pattern ends
            self.set_all_motors(0)
            self.pattern_running = False
            self.send_event("pattern_finished")

    def start_pattern(self, pattern_data):
        """Start executing a vibration pattern"""
        self.stop_pattern()  # Stop any existing pattern

        self.pattern_running = True
        self.pattern_thread = threading.Thread(
            target=self._pattern_worker,
            args=(pattern_data,),
            daemon=True
        )
        self.pattern_thread.start()

    def stop_pattern(self):
        """Stop current pattern execution"""
        if self.pattern_running:
            self.pattern_running = False
            if self.pattern_thread:
                self.pattern_thread.join(timeout=1)
            self.set_all_motors(0)

    def handle_process_command(self, command_data):
        """Handle motor control specific commands"""
        command = command_data.get("command")
        parameters = command_data.get("parameters", {})
        correlation_id = command_data.get("correlation_id")

        try:
            if command == "set_motor":
                # Set individual motor state
                motor_name = parameters.get("motor")
                state = parameters.get("state", 0)

                if not motor_name:
                    return {
                        "status": "error",
                        "error": "Motor name required",
                        "correlation_id": correlation_id
                    }

                success = self.set_motor_state(motor_name, state)
                if success:
                    self.send_event("motor_changed", {
                        "motor": motor_name,
                        "state": state
                    })

                return {
                    "status": "ok" if success else "error",
                    "action": "motor_set",
                    "motor": motor_name,
                    "state": state,
                    "correlation_id": correlation_id
                }

            elif command == "set_all_motors":
                # Set all motors to same state
                state = parameters.get("state", 0)
                self.set_all_motors(state)

                self.send_event("all_motors_changed", {"state": state})

                return {
                    "status": "ok",
                    "action": "all_motors_set",
                    "state": state,
                    "correlation_id": correlation_id
                }

            elif command == "start_pattern":
                # Start vibration pattern
                pattern = parameters.get("pattern", [])
                repeat = parameters.get("repeat", False)

                if not pattern:
                    return {
                        "status": "error",
                        "error": "Pattern data required",
                        "correlation_id": correlation_id
                    }

                self.start_pattern({"pattern": pattern, "repeat": repeat})

                self.send_event("pattern_started", {
                    "pattern_length": len(pattern),
                    "repeat": repeat
                })

                return {
                    "status": "ok",
                    "action": "pattern_started",
                    "correlation_id": correlation_id
                }

            elif command == "stop_pattern":
                # Stop current pattern
                self.stop_pattern()

                return {
                    "status": "ok",
                    "action": "pattern_stopped",
                    "correlation_id": correlation_id
                }

            elif command == "get_motors":
                # Return available motors
                return {
                    "status": "ok",
                    "motors": list(self.motors.keys()),
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
    process = MotorControlProcess()
    try:
        process.run()
    except KeyboardInterrupt:
        print("\nMotor Control Process: Received interrupt")
    finally:
        process.cleanup()