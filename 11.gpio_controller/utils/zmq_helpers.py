#!/usr/bin/env python3
"""
ZeroMQ helper utilities
"""
import zmq
import json
import time
import uuid

class ZMQClient:
    """Helper class for ZeroMQ client operations"""

    def __init__(self, server_address, timeout=5000):
        """
        Initialize ZMQ client

        Args:
            server_address: Server address (e.g., "tcp://localhost:5558")
            timeout: Timeout in milliseconds
        """
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.setsockopt(zmq.RCVTIMEO, timeout)
        self.socket.connect(server_address)

    def send_command(self, command, parameters=None):
        """
        Send command to server

        Args:
            command: Command string
            parameters: Optional parameters dictionary

        Returns:
            Response dictionary or None if timeout
        """
        command_data = {
            "command": command,
            "parameters": parameters or {},
            "timestamp": time.time(),
            "correlation_id": str(uuid.uuid4())
        }

        try:
            self.socket.send_string(json.dumps(command_data))
            response = self.socket.recv_string()
            return json.loads(response)
        except zmq.Again:
            return {"status": "timeout", "error": "Command timeout"}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def close(self):
        """Close the client connection"""
        self.socket.close()
        self.context.term()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

class ZMQSubscriber:
    """Helper class for ZeroMQ subscriber operations"""

    def __init__(self, server_addresses, topic_filter=""):
        """
        Initialize ZMQ subscriber

        Args:
            server_addresses: List of server addresses or single address
            topic_filter: Topic filter string (empty for all messages)
        """
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)

        # Connect to servers
        if isinstance(server_addresses, str):
            server_addresses = [server_addresses]

        for address in server_addresses:
            self.socket.connect(address)

        # Set topic filter
        self.socket.setsockopt_string(zmq.SUBSCRIBE, topic_filter)

    def receive_event(self, timeout=100):
        """
        Receive event from publishers

        Args:
            timeout: Timeout in milliseconds

        Returns:
            Event dictionary or None if timeout
        """
        try:
            if self.socket.poll(timeout):
                message = self.socket.recv_string(zmq.NOBLOCK)
                return json.loads(message)
        except zmq.Again:
            pass
        except Exception as e:
            print(f"Error receiving event: {e}")
        return None

    def close(self):
        """Close the subscriber connection"""
        self.socket.close()
        self.context.term()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

def create_command_client(process_name, host="localhost"):
    """
    Create a command client for a specific process

    Args:
        process_name: Name of the target process
        host: Target host (default: localhost)

    Returns:
        ZMQClient instance
    """
    from ..config.settings import ZeroMQConfig

    port_map = {
        'rgb_light': ZeroMQConfig.RGB_COMMAND_PORT,
        'motor_control': ZeroMQConfig.MOTOR_COMMAND_PORT,
        'input_monitor': ZeroMQConfig.INPUT_COMMAND_PORT
    }

    port = port_map.get(process_name)
    if not port:
        raise ValueError(f"Unknown process: {process_name}")

    return ZMQClient(f"tcp://{host}:{port}")

def create_event_subscriber(process_names=None, host="localhost"):
    """
    Create an event subscriber for specific processes

    Args:
        process_names: List of process names or None for all
        host: Target host (default: localhost)

    Returns:
        ZMQSubscriber instance
    """
    from ..config.settings import ZeroMQConfig

    port_map = {
        'rgb_light': ZeroMQConfig.RGB_EVENT_PORT,
        'motor_control': ZeroMQConfig.MOTOR_EVENT_PORT,
        'input_monitor': ZeroMQConfig.INPUT_EVENT_PORT
    }

    if process_names is None:
        process_names = list(port_map.keys())

    addresses = []
    for process_name in process_names:
        port = port_map.get(process_name)
        if port:
            addresses.append(f"tcp://{host}:{port}")

    return ZMQSubscriber(addresses)