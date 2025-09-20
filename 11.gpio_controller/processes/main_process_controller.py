#!/usr/bin/env python3
import zmq
import json
import time
import signal
import subprocess
import threading
import os
import uuid
from dotenv import load_dotenv

class MainProcessController:
    """Main process controller that coordinates all GPIO worker processes"""

    def __init__(self):
        load_dotenv('config/.env')

        self.running = True
        self.context = zmq.Context()

        # Worker process information
        self.workers = {
            'rgb_light': {
                'script': 'processes/rgb_light_process.py',
                'command_port': int(os.getenv('RGB_COMMAND_PORT', 5558)),
                'event_port': int(os.getenv('RGB_EVENT_PORT', 5559)),
                'process': None,
                'last_seen': time.time()
            },
            'motor_control': {
                'script': 'processes/motor_control_process.py',
                'command_port': int(os.getenv('MOTOR_COMMAND_PORT', 5560)),
                'event_port': int(os.getenv('MOTOR_EVENT_PORT', 5561)),
                'process': None,
                'last_seen': time.time()
            },
            'input_monitor': {
                'script': 'processes/input_monitor_process.py',
                'command_port': int(os.getenv('INPUT_COMMAND_PORT', 5562)),
                'event_port': int(os.getenv('INPUT_EVENT_PORT', 5563)),
                'process': None,
                'last_seen': time.time()
            }
        }

        # ZeroMQ sockets
        self.command_sockets = {}  # REQ sockets to send commands to workers
        self.event_socket = None   # SUB socket to receive events from workers
        self.status_socket = None  # PULL socket to receive status from workers

        # Status collector port
        self.status_port = int(os.getenv('STATUS_COLLECTOR_PORT', 5557))

        # Current system state
        self.current_color = (0.0, 1.0, 0.0)  # Default green
        self.input_states = {'button_1': 0, 'button_2': 0}
        self.last_input_change = time.time()

        # Color definitions
        self.colors = {
            'green': (0.0, 1.0, 0.0),      # Default
            'red': (1.0, 0.0, 0.0),        # Button 1
            'blue': (0.0, 0.0, 1.0),       # Button 2
            'white': (1.0, 1.0, 1.0),      # Both buttons
        }

        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, _sig, _frame):
        """Handle shutdown signals"""
        print("\nMain Controller: Received shutdown signal")
        self.shutdown()

    def initialize_sockets(self):
        """Initialize ZeroMQ sockets for communication with workers"""
        try:
            # Create REQ sockets for sending commands to each worker
            for worker_name, worker_info in self.workers.items():
                socket = self.context.socket(zmq.REQ)
                socket.connect(f"tcp://localhost:{worker_info['command_port']}")
                # Set timeout for commands
                socket.setsockopt(zmq.RCVTIMEO, 5000)  # 5 second timeout
                self.command_sockets[worker_name] = socket
                print(f"Main Controller: Connected to {worker_name} command port {worker_info['command_port']}")

            # Create SUB socket for receiving events from all workers
            self.event_socket = self.context.socket(zmq.SUB)
            for worker_name, worker_info in self.workers.items():
                self.event_socket.connect(f"tcp://localhost:{worker_info['event_port']}")
            self.event_socket.setsockopt_string(zmq.SUBSCRIBE, "")  # Subscribe to all events
            print("Main Controller: Connected to all worker event ports")

            # Create PULL socket for receiving status updates
            self.status_socket = self.context.socket(zmq.PULL)
            self.status_socket.bind(f"tcp://*:{self.status_port}")
            print(f"Main Controller: Status collector bound to port {self.status_port}")

        except Exception as e:
            print(f"Main Controller: Error initializing sockets: {e}")
            raise

    def start_worker_processes(self):
        """Start all worker processes"""
        for worker_name, worker_info in self.workers.items():
            try:
                print(f"Main Controller: Starting {worker_name} process")
                process = subprocess.Popen(['python3', worker_info['script']])
                worker_info['process'] = process
                worker_info['last_seen'] = time.time()
                time.sleep(1)  # Give process time to start
            except Exception as e:
                print(f"Main Controller: Error starting {worker_name}: {e}")

    def stop_worker_processes(self):
        """Stop all worker processes"""
        # First try to send shutdown commands
        for worker_name in self.workers:
            try:
                self.send_command(worker_name, "shutdown")
            except Exception as e:
                print(f"Main Controller: Error sending shutdown to {worker_name}: {e}")

        # Wait a bit for graceful shutdown
        time.sleep(2)

        # Force terminate if still running
        for worker_name, worker_info in self.workers.items():
            process = worker_info['process']
            if process and process.poll() is None:
                print(f"Main Controller: Force terminating {worker_name}")
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()

    def send_command(self, worker_name, command, parameters=None):
        """Send command to a specific worker"""
        if worker_name not in self.command_sockets:
            raise ValueError(f"Unknown worker: {worker_name}")

        command_data = {
            "command": command,
            "parameters": parameters or {},
            "timestamp": time.time(),
            "correlation_id": str(uuid.uuid4())
        }

        try:
            socket = self.command_sockets[worker_name]
            socket.send_string(json.dumps(command_data))
            response = socket.recv_string()
            return json.loads(response)
        except zmq.Again:
            return {"status": "timeout", "error": "Command timeout"}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def handle_input_event(self, event_data):
        """Handle input events from input monitor"""
        event_type = event_data.get("event")
        data = event_data.get("data", {})

        if event_type == "input_status":
            # Update current input states
            new_states = data.get("states", {})
            if new_states != self.input_states:
                self.input_states = new_states
                self.last_input_change = time.time()
                self.update_system_state()

        elif event_type == "button_pressed":
            button = data.get("button")
            print(f"Main Controller: Button {button} pressed")

        elif event_type == "button_released":
            button = data.get("button")
            print(f"Main Controller: Button {button} released")

    def update_system_state(self):
        """Update system state based on current inputs"""
        active_inputs = sum(self.input_states.values())
        new_color = self.current_color
        vibration_pattern = None

        # Determine new color and vibration pattern based on inputs
        if active_inputs == 0:
            new_color = self.colors['green']  # Default
        elif active_inputs == 2:
            # Both buttons pressed
            new_color = self.colors['white']
            vibration_pattern = [
                {"motor": "all", "state": 1, "duration": 0.5},
                {"motor": "all", "state": 0, "duration": 0.5}
            ]
        elif self.input_states.get('button_1', 0):
            # Button 1 only
            new_color = self.colors['red']
            vibration_pattern = [
                {"motor": "motor_1", "state": 1, "duration": 0.3},
                {"motor": "motor_1", "state": 0, "duration": 0.3}
            ]
        elif self.input_states.get('button_2', 0):
            # Button 2 only
            new_color = self.colors['blue']
            vibration_pattern = [
                {"motor": "motor_2", "state": 1, "duration": 0.3},
                {"motor": "motor_2", "state": 0, "duration": 0.3}
            ]

        # Update RGB light if color changed
        if new_color != self.current_color:
            self.current_color = new_color
            color_name = next((name for name, rgb in self.colors.items() if rgb == new_color), 'unknown')
            print(f"Main Controller: Switching to {color_name} breathing light")

            # Send command to RGB light process
            self.send_command('rgb_light', 'start_breathing', {
                'color': list(new_color),
                'duration': 2.5
            })

        # Update vibration motors if pattern specified
        if vibration_pattern:
            self.send_command('motor_control', 'start_pattern', {
                'pattern': vibration_pattern,
                'repeat': False
            })

    def monitor_events(self):
        """Monitor events from all worker processes"""
        while self.running:
            try:
                if self.event_socket.poll(timeout=100):  # 100ms timeout
                    message = self.event_socket.recv_string(zmq.NOBLOCK)
                    event_data = json.loads(message)

                    source = event_data.get("source")
                    event_type = event_data.get("event")

                    # Handle events based on source
                    if source == "input_monitor":
                        self.handle_input_event(event_data)
                    else:
                        # Log other events
                        print(f"Main Controller: Event from {source}: {event_type}")

            except zmq.Again:
                pass
            except Exception as e:
                print(f"Main Controller: Error monitoring events: {e}")
                time.sleep(0.1)

    def monitor_status(self):
        """Monitor status updates from worker processes"""
        while self.running:
            try:
                if self.status_socket.poll(timeout=100):  # 100ms timeout
                    message = self.status_socket.recv_string(zmq.NOBLOCK)
                    status_data = json.loads(message)

                    process_name = status_data.get("process")
                    status = status_data.get("status")

                    if process_name in self.workers:
                        self.workers[process_name]['last_seen'] = time.time()

                    # Log status changes
                    if status != "running":
                        print(f"Main Controller: {process_name} status: {status}")

            except zmq.Again:
                pass
            except Exception as e:
                print(f"Main Controller: Error monitoring status: {e}")
                time.sleep(0.1)

    def run(self):
        """Main controller run loop"""
        try:
            print("Main Controller: Starting GPIO process controller")

            # Start worker processes
            self.start_worker_processes()

            # Wait for workers to initialize
            time.sleep(3)

            # Initialize communication
            self.initialize_sockets()

            # Start event monitoring thread
            event_thread = threading.Thread(target=self.monitor_events, daemon=True)
            event_thread.start()

            # Start status monitoring thread
            status_thread = threading.Thread(target=self.monitor_status, daemon=True)
            status_thread.start()

            # Initialize system with default state
            print("Main Controller: Initializing system state")
            self.send_command('rgb_light', 'start_breathing', {
                'color': list(self.current_color),
                'duration': 2.5
            })

            print("Main Controller: System ready - monitoring and controlling GPIO processes")
            print("Press Ctrl+C to shutdown")

            # Main loop - keep alive and handle any main controller logic
            while self.running:
                time.sleep(1)

        except Exception as e:
            print(f"Main Controller: Fatal error: {e}")
        finally:
            self.shutdown()

    def shutdown(self):
        """Shutdown the main controller and all workers"""
        print("Main Controller: Shutting down")
        self.running = False

        # Stop worker processes
        self.stop_worker_processes()

        # Close sockets
        for socket in self.command_sockets.values():
            socket.close()
        if self.event_socket:
            self.event_socket.close()
        if self.status_socket:
            self.status_socket.close()

        # Terminate context
        self.context.term()

        print("Main Controller: Shutdown complete")


if __name__ == "__main__":
    controller = MainProcessController()
    try:
        controller.run()
    except KeyboardInterrupt:
        print("\nMain Controller: Received keyboard interrupt")
    finally:
        controller.shutdown()