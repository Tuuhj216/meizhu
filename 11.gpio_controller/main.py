#!/usr/bin/env python3
from gpio.gpio_controller import GPIOController

def main():
    print("Starting NXP GPIO Controller...")

    controller = GPIOController()

    try:
        controller.initialize_gpio()
        controller.run_control_loop()
    except Exception as e:
        print(f"Error: {e}")
    finally:
        controller.cleanup()

    print("GPIO Controller shutdown complete")

if __name__ == "__main__":
    main()