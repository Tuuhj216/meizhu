#!/usr/bin/env python3
import gpiod
import time
import math
import signal
import sys

class BreathingLight:
    def __init__(self, chip_name='gpiochip0', pin=23):
        self.chip = gpiod.Chip(chip_name)
        self.line = self.chip.get_line(pin)
        self.line.request(consumer="breathing_light", type=gpiod.LINE_REQ_DIR_OUT)
        self.running = True
        
        # Setup signal handler for clean exit
        signal.signal(signal.SIGINT, self.signal_handler)
    
    def signal_handler(self, sig, frame):
        print("\nExiting gracefully...")
        self.running = False
    
    def software_pwm(self, duty_cycle, frequency=1000):
        """
        Software PWM implementation
        duty_cycle: 0.0 to 1.0 (0% to 100%)
        frequency: PWM frequency in Hz
        """
        period = 1.0 / frequency
        on_time = period * duty_cycle
        off_time = period * (1 - duty_cycle)
        
        if duty_cycle > 0:
            self.line.set_value(1)
            time.sleep(on_time)
        
        if duty_cycle < 1:
            self.line.set_value(0)
            time.sleep(off_time)
    
    def breathing_effect_smooth(self, duration=2.0, steps=100):
        """Smooth breathing effect using sine wave"""
        print("Starting smooth breathing effect (Ctrl+C to stop)")
        
        while self.running:
            for i in range(steps):
                if not self.running:
                    break
                
                # Create sine wave from 0 to 1
                angle = (i / steps) * 2 * math.pi
                brightness = (math.sin(angle) + 1) / 2  # 0 to 1
                
                # Apply PWM for this brightness level
                start_time = time.time()
                step_duration = duration / steps
                
                while time.time() - start_time < step_duration:
                    if not self.running:
                        break
                    self.software_pwm(brightness, frequency=200)
    
    def breathing_effect_linear(self, duration=2.0, steps=50):
        """Linear fade in/out breathing effect"""
        print("Starting linear breathing effect (Ctrl+C to stop)")
        
        while self.running:
            # Fade in
            for i in range(steps):
                if not self.running:
                    break
                brightness = i / steps
                self.apply_brightness(brightness, duration / (steps * 2))
            
            # Fade out
            for i in range(steps, 0, -1):
                if not self.running:
                    break
                brightness = i / steps
                self.apply_brightness(brightness, duration / (steps * 2))
    
    def apply_brightness(self, brightness, step_time):
        """Apply brightness level for given time"""
        start_time = time.time()
        while time.time() - start_time < step_time:
            if not self.running:
                break
            self.software_pwm(brightness, frequency=200)
    
    def breathing_effect_exponential(self, duration=2.0):
        """Exponential breathing effect (more natural looking)"""
        print("Starting exponential breathing effect (Ctrl+C to stop)")
        
        while self.running:
            start_time = time.time()
            
            while time.time() - start_time < duration:
                if not self.running:
                    break
                
                # Calculate position in cycle (0 to 1)
                progress = (time.time() - start_time) / duration
                
                # Create exponential breathing curve
                if progress < 0.5:
                    # Breathing in - exponential rise
                    brightness = math.pow(progress * 2, 2)
                else:
                    # Breathing out - exponential fall
                    brightness = math.pow((1 - progress) * 2, 2)
                
                self.software_pwm(brightness, frequency=200)
    
    def simple_blink_fade(self, on_time=1.0, off_time=1.0, fade_steps=20):
        """Simple fade in/out with adjustable timing"""
        print("Starting simple fade breathing (Ctrl+C to stop)")
        
        while self.running:
            # Fade in
            for i in range(fade_steps):
                if not self.running:
                    break
                brightness = i / fade_steps
                self.apply_brightness(brightness, on_time / fade_steps)
            
            # Stay on briefly
            time.sleep(0.1)
            
            # Fade out
            for i in range(fade_steps, 0, -1):
                if not self.running:
                    break
                brightness = i / fade_steps
                self.apply_brightness(brightness, off_time / fade_steps)
            
            # Stay off briefly
            time.sleep(0.1)
    
    def cleanup(self):
        """Clean up GPIO resources"""
        self.line.set_value(0)  # Turn off LED
        self.line.release()
        self.chip.close()
        print("GPIO cleaned up")