#!/usr/bin/env python3
"""
GPIO Information Reader - Access GPIO details without root privileges
Uses libgpiod for modern GPIO access
"""

import gpiod
import os
import sys

def get_gpio_chip_info(chip_path="/dev/gpiochip4"):
    """Get detailed information about a GPIO chip"""
    try:
        chip = gpiod.Chip(chip_path)
        print(f"Chip: {chip_path}")
        print(f"Name: {chip.name}")
        print(f"Label: {chip.label}")
        print(f"Lines: {chip.num_lines}")
        print("-" * 50)

        # Get line information
        for line_num in range(chip.num_lines):
            line = chip.get_line(line_num)
            info = line.info()

            # Format line information
            used = "[USED]" if info.is_used else "[FREE]"
            direction = "OUT" if info.direction == gpiod.Line.DIRECTION_OUTPUT else "IN "
            active_state = "ACTIVE_LOW" if info.active_state == gpiod.Line.ACTIVE_LOW else "ACTIVE_HIGH"
            bias = ""
            if hasattr(info, 'bias'):
                if info.bias == gpiod.Line.BIAS_PULL_UP:
                    bias = " PULL-UP"
                elif info.bias == gpiod.Line.BIAS_PULL_DOWN:
                    bias = " PULL-DOWN"

            consumer = f" ({info.consumer})" if info.consumer else ""
            name = info.name if info.name else "unnamed"

            print(f"Line {line_num:2d}: {name:15} {used} {direction} {active_state}{bias}{consumer}")

        chip.close()

    except Exception as e:
        print(f"Error accessing {chip_path}: {e}")

def monitor_button_states():
    """Monitor the two user buttons in real-time"""
    try:
        chip = gpiod.Chip("/dev/gpiochip4")

        # Get button lines (5 and 6 based on gpioinfo output)
        button1_line = chip.get_line(5)  # User Button1
        button2_line = chip.get_line(6)  # User Button2

        # Request lines for input
        button1_line.request(consumer="button_monitor", type=gpiod.LINE_REQ_DIR_IN)
        button2_line.request(consumer="button_monitor", type=gpiod.LINE_REQ_DIR_IN)

        print("Monitoring buttons (Ctrl+C to exit):")
        print("Button1 (line 5) | Button2 (line 6)")
        print("-" * 35)

        try:
            while True:
                btn1_val = button1_line.get_value()
                btn2_val = button2_line.get_value()
                print(f"\r     {btn1_val}        |      {btn2_val}     ", end="", flush=True)

        except KeyboardInterrupt:
            print("\nMonitoring stopped.")

        finally:
            button1_line.release()
            button2_line.release()
            chip.close()

    except Exception as e:
        print(f"Error monitoring buttons: {e}")

def get_sysfs_gpio_info():
    """Get GPIO information from sysfs (alternative method)"""
    gpio_base_path = "/sys/devices/platform/soc@0/44000000.bus/44350000.i2c/i2c-1/1-0022/gpiochip4"

    if os.path.exists(gpio_base_path):
        try:
            with open(f"{gpio_base_path}/base", "r") as f:
                base = f.read().strip()
            with open(f"{gpio_base_path}/ngpio", "r") as f:
                ngpio = f.read().strip()
            with open(f"{gpio_base_path}/label", "r") as f:
                label = f.read().strip()

            print(f"Sysfs GPIO Info:")
            print(f"Base: {base}")
            print(f"Number of GPIOs: {ngpio}")
            print(f"Label: {label}")

        except Exception as e:
            print(f"Error reading sysfs GPIO info: {e}")
    else:
        print("Sysfs GPIO path not found")

def main():
    print("=== GPIO Information Tool ===\n")

    if len(sys.argv) > 1 and sys.argv[1] == "monitor":
        monitor_button_states()
    else:
        # Show chip information
        get_gpio_chip_info()
        print()

        # Show sysfs information
        get_sysfs_gpio_info()

        print("\nTo monitor button states in real-time, run:")
        print("python3 gpio_info.py monitor")

if __name__ == "__main__":
    main()