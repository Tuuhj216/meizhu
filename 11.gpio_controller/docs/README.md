# Distributed GPIO Controller System

A modular, process-based GPIO controller system using ZeroMQ for inter-process communication.

## Architecture

The system is built with a distributed architecture where different GPIO functionalities run in separate processes:

```
Main Controller (Coordinator)
├── RGB Light Process      (RGB LED control)
├── Motor Control Process  (Vibration motors)
└── Input Monitor Process  (Button monitoring)
```

## Features

- **Process Isolation**: Each component runs in a separate process for fault tolerance
- **Hot Restart**: Individual processes can be restarted without affecting others
- **Remote Control**: ZeroMQ enables control from remote machines
- **Event-Driven**: Real-time event broadcasting between processes
- **Configurable**: All pins and ports configurable via `.env` file
- **Monitoring**: Built-in health monitoring and status reporting

## File Structure

### Core Components
- `single_gpio_controller.py` - Single pin GPIO controller class
- `rgb_light_controller.py` - RGB light controller using single GPIO controllers
- `gpio_controller.py` - Original monolithic controller (for reference)

### Process Architecture
- `base_process.py` - Base class for all worker processes
- `rgb_light_process.py` - RGB light control process
- `motor_control_process.py` - Vibration motor control process
- `input_monitor_process.py` - Input button monitoring process
- `main_process_controller.py` - Main coordinator process

### Configuration
- `.env` - GPIO pin and port configuration
- `requirements.txt` - Python dependencies

### Documentation
- `ARCHITECTURE.md` - Detailed architecture documentation
- `README.md` - This file

### Utilities
- `start_gpio_system.py` - System startup script

## Installation

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. Install GPIO library:
```bash
sudo apt-get install python3-libgpiod
```

3. Configure GPIO pins in `.env` file (if needed)

## Usage

### Quick Start
```bash
python3 start_gpio_system.py
```

### Manual Process Control

Start individual processes:
```bash
# Terminal 1 - RGB Light Process
python3 rgb_light_process.py

# Terminal 2 - Motor Control Process
python3 motor_control_process.py

# Terminal 3 - Input Monitor Process
python3 input_monitor_process.py

# Terminal 4 - Main Controller
python3 main_process_controller.py
```

### Remote Control

Send commands to processes via ZeroMQ:

```python
import zmq
import json

context = zmq.Context()
socket = context.socket(zmq.REQ)
socket.connect("tcp://localhost:5558")  # RGB light command port

# Set RGB color
command = {
    "command": "set_color",
    "parameters": {"red": 1, "green": 0, "blue": 0}
}
socket.send_string(json.dumps(command))
response = socket.recv_string()
print(json.loads(response))
```

## Configuration

### GPIO Pins (.env)
```bash
# RGB LED Pins
RGB_RED_PIN=27
RGB_GREEN_PIN=17
RGB_BLUE_PIN=22

# Motor Pins
VIBRATION_MOTOR_1_PIN=23
VIBRATION_MOTOR_2_PIN=24

# Input Pins
BUTTON_1_PIN=5
BUTTON_2_PIN=6

# GPIO Chips
OUTPUT_CHIP=gpiochip0
INPUT_CHIP=gpiochip4
```

### ZeroMQ Ports (.env)
```bash
RGB_COMMAND_PORT=5558
RGB_EVENT_PORT=5559
MOTOR_COMMAND_PORT=5560
MOTOR_EVENT_PORT=5561
INPUT_COMMAND_PORT=5562
INPUT_EVENT_PORT=5563
STATUS_COLLECTOR_PORT=5557
```

## Command Reference

### RGB Light Commands
- `set_color` - Set static RGB color
- `start_breathing` - Start breathing effect
- `stop_effect` - Stop current effect
- `turn_off` - Turn off all LEDs

### Motor Control Commands
- `set_motor` - Control individual motor
- `set_all_motors` - Control all motors
- `start_pattern` - Execute vibration pattern
- `stop_pattern` - Stop current pattern

### Input Monitor Commands
- `get_states` - Get current button states
- `get_inputs` - List available inputs
- `set_debounce` - Set debounce time

## System Behavior

### Input Response
- **No buttons**: Green breathing light
- **Button 1**: Red breathing light + Motor 1 vibration
- **Button 2**: Blue breathing light + Motor 2 vibration
- **Both buttons**: White breathing light + Both motors vibration

### Events
The system broadcasts events for:
- Button press/release
- Color changes
- Motor state changes
- Process status updates

## Troubleshooting

### Check Process Status
```bash
ps aux | grep python3
```

### Check ZeroMQ Ports
```bash
netstat -ln | grep 555
```

### Debug Individual Processes
Run individual processes with verbose output to debug issues.

### Permission Issues
Ensure your user has GPIO access:
```bash
sudo usermod -a -G gpio $USER
```

## Benefits Over Monolithic Design

1. **Fault Isolation**: Process crash doesn't affect entire system
2. **Performance**: Multi-core utilization with separate processes
3. **Scalability**: Easy to add new GPIO controllers
4. **Remote Control**: Network-accessible via ZeroMQ
5. **Hot Restart**: Restart components without full shutdown
6. **Monitoring**: Real-time status and health checking
7. **Testing**: Individual components can be tested in isolation