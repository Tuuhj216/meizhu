#!/usr/bin/env python3
import os
import time
import random
from .base_process import BaseProcess
from ..gpio.single_gpio_controller import SingleGPIOController

class InputMonitorProcess(BaseProcess):
    """Input monitoring process for button inputs"""

    def __init__(self):
        super().__init__(
            process_name="input_monitor",
            command_port=int(os.getenv('INPUT_COMMAND_PORT', 5562)),
            event_port=int(os.getenv('INPUT_EVENT_PORT', 5563)),
        )
        self.input_controllers = {}
        self.last_states = {}
        self.debounce_time = 0.05  # 50ms debounce

    def initialize_hardware(self):
        """Initialize input pin controllers"""
        try:
            chip_name = os.getenv('INPUT_CHIP', 'gpiochip4')

            # Load input pin configuration from environment
            input_pins = {
                'button_1': int(os.getenv('BUTTON_1_PIN', 5)),
                'button_2': int(os.getenv('BUTTON_2_PIN', 6))
            }

            # Initialize each input controller
            for input_name, pin in input_pins.items():
                try:
                    controller = SingleGPIOController(chip_name, pin, 'in', f'input_{input_name}')
                    self.input_controllers[input_name] = controller
                    self.last_states[input_name] = 0  # Initialize as not pressed
                    print(f"{self.process_name}: Initialized {input_name} on pin {pin}")
                except Exception as e:
                    print(f"{self.process_name}: Warning - Could not initialize {input_name}: {e}")

            if not self.input_controllers:
                print(f"{self.process_name}: No input controllers initialized - using system polling")

            self.send_event("inputs_initialized", {
                "inputs": list(self.input_controllers.keys()),
                "chip": chip_name,
                "fallback_mode": len(self.input_controllers) == 0
            })

        except Exception as e:
            print(f"{self.process_name}: Error initializing input hardware: {e}")
            raise

    def cleanup_hardware(self):
        """Clean up input controllers"""
        for input_name, controller in self.input_controllers.items():
            try:
                controller.cleanup()
                print(f"{self.process_name}: Cleaned up {input_name}")
            except Exception as e:
                print(f"{self.process_name}: Error cleaning up {input_name}: {e}")

    def read_input_states_direct(self):
        """Read input states directly from GPIO controllers"""
        states = {}
        for name, controller in self.input_controllers.items():
            try:
                states[name] = controller.get_value()
            except Exception as e:
                print(f"{self.process_name}: Error reading {name}: {e}")
                states[name] = 0
        return states

    def read_input_states_system(self):
        """Read input states using system polling method (fallback)"""
        states = {}
        try:
            # Method 1: Read from GPIO debug info (most reliable)
            import subprocess
            try:
                result = subprocess.run(['sudo', 'cat', '/sys/kernel/debug/gpio'],
                                      capture_output=True, text=True, timeout=2)
                if result.returncode == 0:
                    gpio_info = result.stdout
                    # Parse gpiochip4 section for User Button states
                    for line in gpio_info.split('\n'):
                        if 'User Button1' in line and 'gpio-645' in line:
                            # Extract state: look for 'in  hi' or 'in  lo'
                            states['button_1'] = 1 if ' hi ' in line else 0
                        elif 'User Button2' in line and 'gpio-646' in line:
                            states['button_2'] = 1 if ' hi ' in line else 0

                    # If we got both buttons from debug info, return
                    if 'button_1' in states and 'button_2' in states:
                        return states
            except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
                pass  # Fall through to other methods

            # Method 2: Try reading from /sys/class/gpio if exported
            for name, pin in [('button_1', 5), ('button_2', 6)]:
                if name not in states:  # Only if not already read from debug
                    gpio_path = f"/sys/class/gpio/gpio{pin}/value"
                    if os.path.exists(gpio_path):
                        with open(gpio_path, 'r') as f:
                            states[name] = int(f.read().strip())
                    else:
                        # Simulate random input for testing if hardware not available
                        states[name] = random.choice([0, 1]) if random.random() < 0.05 else 0

        except Exception as e:
            print(f"{self.process_name}: Error in system polling: {e}")
            # Default all inputs to 0
            states = {'button_1': 0, 'button_2': 0}

        return states

    def get_input_states(self):
        """Get current input states using best available method"""
        if self.input_controllers:
            return self.read_input_states_direct()
        else:
            return self.read_input_states_system()

    def monitor_inputs(self):
        """Main input monitoring loop"""
        while self.running:
            try:
                # Read current input states
                current_states = self.get_input_states()

                # Check for state changes
                for input_name, current_state in current_states.items():
                    last_state = self.last_states.get(input_name, 0)

                    if current_state != last_state:
                        # State change detected
                        print(f"{self.process_name}: {input_name} changed from {last_state} to {current_state}")

                        # Send state change event
                        self.send_event("input_changed", {
                            "input": input_name,
                            "old_state": last_state,
                            "new_state": current_state,
                            "pressed": current_state == 1
                        })

                        # Send specific press/release events
                        if current_state == 1 and last_state == 0:
                            self.send_event("button_pressed", {
                                "button": input_name
                            })
                        elif current_state == 0 and last_state == 1:
                            self.send_event("button_released", {
                                "button": input_name
                            })

                        self.last_states[input_name] = current_state

                # Send periodic status with all input states
                self.send_event("input_status", {
                    "states": current_states,
                    "active_count": sum(current_states.values())
                })

                # Check inputs 20 times per second (50ms interval)
                time.sleep(0.05)

            except Exception as e:
                print(f"{self.process_name}: Error in input monitoring: {e}")
                time.sleep(0.1)

    def handle_process_command(self, command_data):
        """Handle input monitor specific commands"""
        command = command_data.get("command")
        parameters = command_data.get("parameters", {})
        correlation_id = command_data.get("correlation_id")

        try:
            if command == "get_states":
                # Return current input states
                states = self.get_input_states()
                return {
                    "status": "ok",
                    "states": states,
                    "active_count": sum(states.values()),
                    "correlation_id": correlation_id
                }

            elif command == "get_inputs":
                # Return available inputs
                return {
                    "status": "ok",
                    "inputs": list(self.last_states.keys()),
                    "correlation_id": correlation_id
                }

            elif command == "set_debounce":
                # Set debounce time
                debounce_time = parameters.get("time", 0.05)
                self.debounce_time = max(0.001, min(1.0, debounce_time))  # 1ms to 1s

                return {
                    "status": "ok",
                    "action": "debounce_set",
                    "debounce_time": self.debounce_time,
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

    def run(self):
        """Override run to include input monitoring"""
        try:
            print(f"{self.process_name}: Starting input monitor process")

            # Initialize everything
            self.initialize_sockets()
            self.initialize_hardware()

            # Start status monitoring thread
            self.status_thread = threading.Thread(target=self._status_monitor_loop, daemon=True)
            self.status_thread.start()

            print(f"{self.process_name}: Process ready, starting input monitoring")

            # Start input monitoring in main thread
            self.monitor_inputs()

        except Exception as e:
            print(f"{self.process_name}: Fatal error: {e}")
        finally:
            self.cleanup()


if __name__ == "__main__":
    process = InputMonitorProcess()
    try:
        process.run()
    except KeyboardInterrupt:
        print("\nInput Monitor Process: Received interrupt")
    finally:
        process.cleanup()