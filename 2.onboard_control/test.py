import gpiod
import time

def test_all_gpio_chips():
    """Test all available GPIO chips"""
    for chip_num in range(6):  # You showed 0-5 in your earlier output
        try:
            chip = gpiod.Chip(f'gpiochip{chip_num}')
            print(f"✓ gpiochip{chip_num}: {chip.num_lines()} lines, name: {chip.name()}")
            
            # Test getting a line (don't configure it yet)
            if chip.num_lines() > 0:
                line = chip.get_line(0)
                print(f"  Line 0 - Used: {line.is_used()}, Consumer: {line.consumer()}")
            
        except Exception as e:
            print(f"✗ gpiochip{chip_num}: {e}")

def test_user_buttons():
    """Test the user buttons you identified earlier"""
    try:
        # From your gpio debug output: gpio-645 and gpio-646 are user buttons
        chip4 = gpiod.Chip('gpiochip4')
        print(f"\nTesting User Buttons on gpiochip4:")
        
        # gpio-645 is pin 5 on chip4 (645-640=5)
        # gpio-646 is pin 6 on chip4 (646-640=6)
        button1 = chip4.get_line(5)  # User Button1
        button2 = chip4.get_line(6)  # User Button2
        
        print(f"Button1 (pin 5): Used={button1.is_used()}, Consumer={button1.consumer()}")
        print(f"Button2 (pin 6): Used={button2.is_used()}, Consumer={button2.consumer()}")
        
        # Try to configure them as inputs
        button1.request(consumer="test-btn1", type=gpiod.LINE_REQ_DIR_IN)
        button2.request(consumer="test-btn2", type=gpiod.LINE_REQ_DIR_IN)
        
        print("\nReading button states (press buttons to test):")
        for i in range(10):
            btn1_val = button1.get_value()
            btn2_val = button2.get_value()
            print(f"Button1: {btn1_val}, Button2: {btn2_val}")
            time.sleep(0.5)
            
        button1.release()
        button2.release()
        print("✓ Button test completed successfully!")
        
    except Exception as e:
        print(f"✗ Button test failed: {e}")

if __name__ == "__main__":
    print("=== GPIO Chip Status ===")
    test_all_gpio_chips()
    
    print("\n=== User Button Test ===")
    test_user_buttons()
