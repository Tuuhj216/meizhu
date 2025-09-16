#!/usr/bin/env python3
import gpiod
import time
import threading
from breath_light import RGBBreathingLight

# Open GPIO chip (usually gpiochip0)
chip = gpiod.Chip('gpiochip0')

buttonChip = gpiod.Chip('gpiochip4')

# Color definitions
COLORS = {
    'green': (0.0, 1.0, 0.0),      # Default/activated
    'purple': (0.8, 0.0, 1.0),     # Left trigger
    'orange': (1.0, 0.5, 0.0)      # Right trigger
}



def main():
    # Create RGB breathing light
    breathing = RGBBreathingLight(chip_name='gpiochip0', red_pin=27, green_pin=17, blue_pin=22)

    # Setup vibration motors
    vibrationL = chip.get_line(23)
    vibrationR = chip.get_line(24)
    vibrationL.request(consumer="my_script", type=gpiod.LINE_REQ_DIR_OUT)
    vibrationR.request(consumer="my_script", type=gpiod.LINE_REQ_DIR_OUT)

    #left_button = buttonChip.get_line(645)
    #right_button = buttonChip.get_line(646)
    
    #left_button = buttonChip.get_line(5)
    #right_button = buttonChip.get_line(6)

    # Setup trigger input pins (assuming pull-up, active low)
    #left_button.request(consumer="triggers", type=gpiod.LINE_REQ_DIR_IN)
    #right_button.request(consumer="triggers", type=gpiod.LINE_REQ_DIR_IN)

    current_color = 'green'  # Default color
    breathing_thread = None

    def start_breathing(color):
        nonlocal breathing_thread
        if breathing_thread and breathing_thread.is_alive():
            breathing.running = False
            breathing_thread.join(timeout=1)

        breathing.running = True
        breathing_thread = threading.Thread(
            target=breathing.breathing_effect_exponential,
            args=(2.5, COLORS[color])
        )
        breathing_thread.daemon = True
        breathing_thread.start()

    # Start with default green breathing
    start_breathing(current_color)

    try:
        print("RGB Breathing Light Active - Green: default, Purple: left trigger, Orange: right trigger")
        print("Press Ctrl+C to exit")

        while True:
            # Read trigger states (assuming active low with pull-up)
            #left_pressed = left_button.get_value() == 0
            left_pressed = True
            right_pressed = False
            #right_pressed = right_button.get_value() == 0

            new_color = current_color

            if left_pressed and right_pressed:
                # Both triggers - keep current color or default to green
                new_color = 'green'
            elif left_pressed:
                new_color = 'purple'
                vibrationL.set_value(1)  # Activate left vibration
            elif right_pressed:
                new_color = 'orange'
                vibrationR.set_value(1)  # Activate right vibration
            else:
                new_color = 'green'  # Default when no triggers
                vibrationL.set_value(0)  # Turn off vibrations
                vibrationR.set_value(0)

            # Change color if different
            if new_color != current_color:
                current_color = new_color
                print(f"Switching to {current_color} breathing")
                start_breathing(current_color)

            time.sleep(0.1)  # Check triggers 10 times per second

    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        breathing.running = False
        if breathing_thread:
            breathing_thread.join(timeout=1)

        breathing.cleanup()
        vibrationL.set_value(0)
        vibrationR.set_value(0)
        vibrationL.release()
        vibrationR.release()
        #left_button.release()
        #right_button.release()
        chip.close()


if __name__ == "__main__":
    main()
