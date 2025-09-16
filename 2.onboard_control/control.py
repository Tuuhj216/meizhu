#!/usr/bin/env python3
import gpiod
import time

# Open GPIO chip (usually gpiochip0)
chip = gpiod.Chip('gpiochip0')

# Get GPIO line 23
line22 = chip.get_line(22)
line23 = chip.get_line(23)
line24 = chip.get_line(24)


# Configure as output
line22.request(consumer="my_script", type=gpiod.LINE_REQ_DIR_OUT)
line23.request(consumer="my_script", type=gpiod.LINE_REQ_DIR_OUT)
line24.request(consumer="my_script", type=gpiod.LINE_REQ_DIR_OUT)

try:
    # Blink LED
    while True:
        line22.set_value(1)  # Turn on
        line23.set_value(1)  # Turn on
        line24.set_value(1)  # Turn on
        time.sleep(0.5)
        line22.set_value(0)  # Turn off
        line23.set_value(0)  # Turn off
        line24.set_value(0)  # Turn off
        time.sleep(0.5)
finally:
    line22.release()
    line23.release()
    line24.release()
    chip.close()
