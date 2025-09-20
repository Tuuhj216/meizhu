#!/usr/bin/env python3
"""
Process management utilities
"""
import subprocess
import time
import signal
import os

class ProcessManager:
    """Utility class for managing system processes"""

    @staticmethod
    def start_process(script_path, cwd=None):
        """
        Start a Python process

        Args:
            script_path: Path to the Python script
            cwd: Working directory (optional)

        Returns:
            Popen object or None if failed
        """
        try:
            process = subprocess.Popen(
                ['python3', script_path],
                cwd=cwd or os.getcwd(),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            return process
        except Exception as e:
            print(f"Error starting process {script_path}: {e}")
            return None

    @staticmethod
    def stop_process(process, timeout=5):
        """
        Stop a process gracefully, with force if needed

        Args:
            process: Popen object
            timeout: Timeout in seconds for graceful shutdown

        Returns:
            True if stopped successfully, False otherwise
        """
        if process is None or process.poll() is not None:
            return True

        try:
            # Try graceful shutdown first
            process.terminate()
            try:
                process.wait(timeout=timeout)
                return True
            except subprocess.TimeoutExpired:
                # Force kill if graceful shutdown failed
                process.kill()
                process.wait()
                return True
        except Exception as e:
            print(f"Error stopping process: {e}")
            return False

    @staticmethod
    def is_process_running(process):
        """
        Check if a process is still running

        Args:
            process: Popen object

        Returns:
            True if running, False otherwise
        """
        return process is not None and process.poll() is None

    @staticmethod
    def wait_for_port(port, timeout=10):
        """
        Wait for a port to become available

        Args:
            port: Port number to check
            timeout: Timeout in seconds

        Returns:
            True if port is available, False if timeout
        """
        import socket
        start_time = time.time()

        while time.time() - start_time < timeout:
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.settimeout(1)
                    result = s.connect_ex(('localhost', port))
                    if result == 0:
                        return True
            except Exception:
                pass
            time.sleep(0.1)

        return False

    @staticmethod
    def get_process_info(process):
        """
        Get information about a process

        Args:
            process: Popen object

        Returns:
            Dictionary with process information
        """
        if process is None:
            return {"status": "not_started", "pid": None, "returncode": None}

        return {
            "status": "running" if process.poll() is None else "stopped",
            "pid": process.pid,
            "returncode": process.returncode
        }