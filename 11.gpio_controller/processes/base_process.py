#!/usr/bin/env python3
import zmq
import json
import time
import signal
import os
import uuid
import threading
from abc import ABC, abstractmethod
from dotenv import load_dotenv

class BaseProcess(ABC):
    """Base class for all GPIO controller processes"""

    def __init__(self, process_name, command_port=None, event_port=None, status_port=None):
        """
        Initialize base process

        Args:
            process_name: Unique name for this process
            command_port: Port for receiving commands (REP socket)
            event_port: Port for publishing events (PUB socket)
            status_port: Port for sending status updates (PUSH socket)
        """
        load_dotenv('config/.env')

        self.process_name = process_name
        self.running = True
        self.context = zmq.Context()

        # Default ports
        self.command_port = command_port or int(os.getenv(f'{process_name.upper()}_COMMAND_PORT', 5555))
        self.event_port = event_port or int(os.getenv(f'{process_name.upper()}_EVENT_PORT', 5556))
        self.status_port = status_port or int(os.getenv('STATUS_COLLECTOR_PORT', 5557))

        # ZeroMQ sockets
        self.command_socket = None  # REP socket for receiving commands
        self.event_socket = None    # PUB socket for publishing events
        self.status_socket = None   # PUSH socket for status updates

        # Status monitoring
        self.last_heartbeat = time.time()
        self.status_thread = None

        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, _sig, _frame):
        """Handle shutdown signals"""
        print(f"\n{self.process_name}: Received shutdown signal")
        self.shutdown()

    def initialize_sockets(self):
        """Initialize ZeroMQ sockets"""
        try:
            # Command socket (REP) - receives commands from main controller
            self.command_socket = self.context.socket(zmq.REP)
            self.command_socket.bind(f"tcp://*:{self.command_port}")
            print(f"{self.process_name}: Command socket bound to port {self.command_port}")

            # Event socket (PUB) - publishes events to subscribers
            self.event_socket = self.context.socket(zmq.PUB)
            self.event_socket.bind(f"tcp://*:{self.event_port}")
            print(f"{self.process_name}: Event socket bound to port {self.event_port}")

            # Status socket (PUSH) - sends status to status collector
            self.status_socket = self.context.socket(zmq.PUSH)
            self.status_socket.connect(f"tcp://localhost:{self.status_port}")
            print(f"{self.process_name}: Status socket connected to port {self.status_port}")

        except Exception as e:
            print(f"{self.process_name}: Error initializing sockets: {e}")
            raise

    def send_event(self, event_type, data=None):
        """Send event to all subscribers"""
        if self.event_socket:
            event = {
                "event": event_type,
                "source": self.process_name,
                "data": data or {},
                "timestamp": time.time()
            }
            try:
                self.event_socket.send_string(json.dumps(event), zmq.NOBLOCK)
            except zmq.Again:
                print(f"{self.process_name}: Warning - Event queue full, dropping event")

    def send_status(self, status="running", metrics=None):
        """Send status update to status collector"""
        if self.status_socket:
            status_msg = {
                "status": status,
                "process": self.process_name,
                "metrics": metrics or {},
                "timestamp": time.time()
            }
            try:
                self.status_socket.send_string(json.dumps(status_msg), zmq.NOBLOCK)
            except zmq.Again:
                print(f"{self.process_name}: Warning - Status queue full, dropping status")

    def handle_command(self, command_data):
        """
        Handle incoming command

        Args:
            command_data: Parsed command dictionary

        Returns:
            Response dictionary
        """
        command = command_data.get("command")
        correlation_id = command_data.get("correlation_id", str(uuid.uuid4()))

        try:
            # Handle common commands
            if command == "ping":
                return {"status": "ok", "correlation_id": correlation_id}
            elif command == "shutdown":
                self.shutdown()
                return {"status": "shutting_down", "correlation_id": correlation_id}
            elif command == "status":
                return {
                    "status": "running" if self.running else "stopped",
                    "process": self.process_name,
                    "correlation_id": correlation_id
                }
            else:
                # Delegate to process-specific handler
                return self.handle_process_command(command_data)

        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "correlation_id": correlation_id
            }

    @abstractmethod
    def handle_process_command(self, command_data):
        """
        Handle process-specific commands

        Args:
            command_data: Parsed command dictionary

        Returns:
            Response dictionary
        """
        pass

    @abstractmethod
    def initialize_hardware(self):
        """Initialize hardware components"""
        pass

    @abstractmethod
    def cleanup_hardware(self):
        """Clean up hardware resources"""
        pass

    def _status_monitor_loop(self):
        """Background thread for sending periodic status updates"""
        while self.running:
            try:
                self.send_status("running", {"last_heartbeat": time.time()})
                time.sleep(5)  # Send status every 5 seconds
            except Exception as e:
                print(f"{self.process_name}: Error in status monitor: {e}")
                time.sleep(1)

    def run(self):
        """Main process loop"""
        try:
            print(f"{self.process_name}: Starting process")

            # Initialize everything
            self.initialize_sockets()
            self.initialize_hardware()

            # Start status monitoring thread
            self.status_thread = threading.Thread(target=self._status_monitor_loop, daemon=True)
            self.status_thread.start()

            print(f"{self.process_name}: Process ready, entering main loop")

            # Main command processing loop
            while self.running:
                try:
                    # Check for commands (non-blocking)
                    if self.command_socket.poll(timeout=100):  # 100ms timeout
                        message = self.command_socket.recv_string(zmq.NOBLOCK)
                        command_data = json.loads(message)

                        # Handle command
                        response = self.handle_command(command_data)
                        self.command_socket.send_string(json.dumps(response))

                    # Update heartbeat
                    self.last_heartbeat = time.time()

                except zmq.Again:
                    # No message available, continue
                    pass
                except KeyboardInterrupt:
                    print(f"\n{self.process_name}: Received keyboard interrupt")
                    break
                except Exception as e:
                    print(f"{self.process_name}: Error in main loop: {e}")
                    time.sleep(0.1)

        except Exception as e:
            print(f"{self.process_name}: Fatal error: {e}")
        finally:
            self.cleanup()

    def shutdown(self):
        """Shutdown the process"""
        print(f"{self.process_name}: Shutting down")
        self.running = False

    def cleanup(self):
        """Clean up all resources"""
        print(f"{self.process_name}: Cleaning up resources")

        # Send final status
        self.send_status("stopped")

        # Clean up hardware
        try:
            self.cleanup_hardware()
        except Exception as e:
            print(f"{self.process_name}: Error cleaning up hardware: {e}")

        # Close sockets
        if self.command_socket:
            self.command_socket.close()
        if self.event_socket:
            self.event_socket.close()
        if self.status_socket:
            self.status_socket.close()

        # Terminate context
        self.context.term()

        print(f"{self.process_name}: Cleanup complete")