#!/usr/bin/env python3
"""
Startup script for the distributed GPIO controller system
"""
import sys
import os

def check_dependencies():
    """Check if required dependencies are available"""
    try:
        import zmq
        import gpiod
        from dotenv import load_dotenv
        print("✓ All dependencies available")
        return True
    except ImportError as e:
        print(f"✗ Missing dependency: {e}")
        print("\nTo install dependencies:")
        print("pip install pyzmq python-dotenv")
        print("sudo apt-get install python3-libgpiod")
        return False

def main():
    print("GPIO Controller System Startup")
    print("=" * 40)

    # Check dependencies
    if not check_dependencies():
        sys.exit(1)

    # Check if .env file exists
    if not os.path.exists('config/.env'):
        print("✗ .env file not found in config/ folder")
        print("Please create a config/.env file with GPIO pin configuration")
        sys.exit(1)
    else:
        print("✓ Configuration file found")

    print("\nStarting distributed GPIO controller system...")
    print("This will start the main process controller which manages:")
    print("- RGB Light Process")
    print("- Motor Control Process")
    print("- Input Monitor Process")
    print("\nPress Ctrl+C to shutdown all processes")
    print("-" * 40)

    # Import and run main controller
    from processes.main_process_controller import MainProcessController

    controller = MainProcessController()
    try:
        controller.run()
    except KeyboardInterrupt:
        print("\nShutdown requested by user")
    except Exception as e:
        print(f"System error: {e}")
    finally:
        controller.shutdown()

if __name__ == "__main__":
    main()