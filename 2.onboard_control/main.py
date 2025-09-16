import gpiod
import time

def control_gpio_output():
    # Let's use gpiochip0 for this example
    chip = gpiod.Chip('gpiochip0')
    
    # Use pin 0 of gpiochip0 (which is GPIO 512 in global numbering)
    output_line = chip.get_line(25)
    
    # Configure as output
    output_line.request(consumer="python-output", type=gpiod.LINE_REQ_DIR_OUT)
    
    try:
        print("Toggling GPIO output (Ctrl+C to exit):")
        state = 0
        while True:
            output_line.set_value(state)
            print(f"GPIO state: {state}")
            state = 1 - state  # Toggle between 0 and 1
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        output_line.release()

if __name__ == "__main__":
    control_gpio_output()
