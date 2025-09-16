#!/usr/bin/env python3
import gpiod
import time
from breath_light import BreathingLight

# Open GPIO chip (usually gpiochip0)
chip = gpiod.Chip('gpiochip0')


def main():
    # Get GPIO line 22
    breathing = BreathingLight(chip_name='gpiochip0', pin=22)

    vibrationL = chip.get_line(23)
    vibrationR = chip.get_line(24)


    # Configure as output
    vibrationL.request(consumer="my_script", type=gpiod.LINE_REQ_DIR_OUT)
    vibrationR.request(consumer="my_script", type=gpiod.LINE_REQ_DIR_OUT)

    try:
        # Blink LED
        breathing.breathing_effect_exponential(duration=2.5)

    finally:
        breathing.cleanup()
        vibrationL.release()
        vibrationR.release()
        chip.close()


if __name__ == "__main__":
    main()
