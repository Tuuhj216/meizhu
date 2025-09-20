# Distributed GPIO Controller System

A modular, process-based GPIO controller system using ZeroMQ for inter-process communication.

## Project Structure

```
11.gpio_controller/
├── gpio/                     # GPIO hardware control modules
│   ├── __init__.py
│   ├── single_gpio_controller.py    # Single pin GPIO controller
│   ├── rgb_light_controller.py      # RGB LED controller
│   ├── gpio_controller.py           # Original monolithic controller
│   └── breath_light.py             # Legacy breathing light
├── processes/                # Process management modules
│   ├── __init__.py
│   ├── base_process.py              # Base class for all processes
│   ├── rgb_light_process.py         # RGB light control process
│   ├── motor_control_process.py     # Motor control process
│   ├── input_monitor_process.py     # Input monitoring process
│   └── main_process_controller.py   # Main coordinator process
├── config/                   # Configuration files
│   ├── __init__.py
│   ├── .env                         # Environment variables
│   ├── settings.py                  # Configuration classes
│   └── requirements.txt             # Python dependencies
├── utils/                    # Utility modules
│   ├── __init__.py
│   ├── logger.py                    # Logging utilities
│   ├── process_manager.py           # Process management helpers
│   ├── zmq_helpers.py               # ZeroMQ helper functions
│   └── start_gpio_system.py         # Legacy startup script
├── docs/                     # Documentation
│   ├── ARCHITECTURE.md              # Architecture documentation
│   └── README.md                    # This file (moved)
└── start_gpio_system.py             # Main startup script
```

## Architecture

The system uses a distributed architecture where different GPIO functionalities run in separate processes:

```
Main Controller (Coordinator)
├── RGB Light Process      (RGB LED control)
├── Motor Control Process  (Vibration motors)
└── Input Monitor Process  (Button monitoring)
```

## Features

- **Modular Design**: Code organized by functionality
- **Process Isolation**: Each component runs in a separate process for fault tolerance
- **Hot Restart**: Individual processes can be restarted without affecting others
- **Remote Control**: ZeroMQ enables control from remote machines
- **Event-Driven**: Real-time event broadcasting between processes
- **Configurable**: All pins and ports configurable via `.env` file
- **Monitoring**: Built-in health monitoring and status reporting
- **Logging**: Structured logging with file and console output

## Installation

1. Install Python dependencies:
```bash
pip install -r config/requirements.txt
```

2. Install GPIO library:
```bash
sudo apt-get install python3-libgpiod
```

3. Configure GPIO pins in `config/.env` file (if needed)

## Usage

### Quick Start
```bash
python3 start_gpio_system.py
```

### Using Configuration Classes
```python
from config.settings import GPIOConfig, ZeroMQConfig

# Access GPIO pin configuration
red_pin = GPIOConfig.RGB_RED_PIN
command_port = ZeroMQConfig.RGB_COMMAND_PORT
```

### Using ZeroMQ Helpers
```python
from utils.zmq_helpers import create_command_client, create_event_subscriber

# Send command to RGB light process
with create_command_client('rgb_light') as client:
    response = client.send_command('set_color', {'red': 1, 'green': 0, 'blue': 0})
    print(response)

# Subscribe to events
with create_event_subscriber(['input_monitor']) as subscriber:
    while True:
        event = subscriber.receive_event(timeout=1000)
        if event:
            print(f"Received event: {event}")
```

### Manual Process Control

Start individual processes:
```bash
# From project root directory
python3 -m processes.rgb_light_process
python3 -m processes.motor_control_process
python3 -m processes.input_monitor_process
python3 -m processes.main_process_controller
```

## Configuration

### GPIO Pins (config/.env)
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

### ZeroMQ Ports (config/.env)
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

## Development

### Adding New Processes
1. Create new process class inheriting from `BaseProcess`
2. Implement required abstract methods
3. Add configuration to `config/settings.py`
4. Update main process controller to include new process

### Logging
```python
from utils.logger import get_process_logger

logger = get_process_logger('my_process')
logger.info("Process started")
```

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

1. **Code Organization**: Clear separation of concerns by functionality
2. **Fault Isolation**: Process crash doesn't affect entire system
3. **Performance**: Multi-core utilization with separate processes
4. **Scalability**: Easy to add new GPIO controllers
5. **Remote Control**: Network-accessible via ZeroMQ
6. **Hot Restart**: Restart components without full shutdown
7. **Monitoring**: Real-time status and health checking
8. **Testing**: Individual components can be tested in isolation
9. **Maintainability**: Modular code is easier to understand and modify