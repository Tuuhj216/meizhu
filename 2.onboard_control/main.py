import gpiod
from gpiod.line import Direction, Value
import time

# GPIO configuration
CHIP_PATH = '/dev/gpiochip2'  # Adjust if needed (e.g., gpiochip0 for other ports)
LINE_OFFSET = 25  # GPIO2_IO17 (change based on your pin; check gpioinfo)
CONSUMER = 'led-test'  # Label for the request

# Request the GPIO line as output
with gpiod.request_lines(
    CHIP_PATH,
    consumer=CONSUMER,
    config={
        LINE_OFFSET: gpiod.LineSettings(
            direction=Direction.OUTPUT,
            output_value=Value.INACTIVE  # Start with LED off
        )
    }
) as request:
    try:
        while True:
            print("LED ON")
            request.set_value(LINE_OFFSET, Value.ACTIVE)  # Turn LED on
            time.sleep(1)
            print("LED OFF")
            request.set_value(LINE_OFFSET, Value.INACTIVE)  # Turn LED off
            time.sleep(1)
    except KeyboardInterrupt:
        print("Program terminated")

# No explicit cleanup needed; context manager handles it