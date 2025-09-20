# GPIO Controller Process Architecture

## Overview
This architecture separates GPIO control into multiple isolated processes that communicate via ZeroMQ messaging patterns.

## Process Structure

```
┌─────────────────────┐
│   Main Controller   │
│   (Coordinator)     │
└─────────┬───────────┘
          │ ZeroMQ
    ┌─────┴─────┐
    │           │
┌───▼───┐   ┌───▼────┐   ┌──────────┐   ┌─────────────┐
│ RGB   │   │ Motor  │   │  Input   │   │   System    │
│ Light │   │Control │   │ Monitor  │   │  Monitor    │
│Process│   │Process │   │ Process  │   │   Process   │
└───────┘   └────────┘   └──────────┘   └─────────────┘
```

## Communication Patterns

### 1. Command & Control (REQ/REP)
- Main controller sends commands to workers
- Workers respond with acknowledgment or status
- Synchronous communication for critical operations

### 2. Event Broadcasting (PUB/SUB)
- Input events broadcast to all interested processes
- State changes published to subscribers
- Asynchronous notifications

### 3. Status Monitoring (PULL/PUSH)
- Health checks and status reports
- Performance metrics collection
- Error reporting

## Process Responsibilities

### Main Controller Process
- User interface and main control logic
- Process lifecycle management (start/stop/restart)
- Command routing and coordination
- Configuration management

### RGB Light Process
- RGB LED control and effects
- Breathing patterns and color changes
- Hardware PWM management

### Motor Control Process
- Vibration motor control
- Pattern execution
- Timing coordination

### Input Monitor Process
- Button state monitoring
- Event generation and broadcasting
- Debouncing and filtering

### System Monitor Process
- Health checks for all processes
- Resource monitoring
- Error detection and reporting
- Automatic restart capabilities

## Message Protocol

### Command Messages
```json
{
  "command": "set_color",
  "target": "rgb_light",
  "parameters": {"red": 1.0, "green": 0.0, "blue": 0.0},
  "timestamp": 1234567890,
  "correlation_id": "uuid"
}
```

### Event Messages
```json
{
  "event": "button_pressed",
  "source": "input_monitor",
  "data": {"button": "button_1", "state": 1},
  "timestamp": 1234567890
}
```

### Status Messages
```json
{
  "status": "running",
  "process": "rgb_light",
  "metrics": {"cpu": 2.5, "memory": 1024},
  "timestamp": 1234567890
}
```

## Configuration

All processes share the same `.env` configuration file but load only relevant settings for their specific functionality.

## Benefits

1. **Fault Isolation**: Individual process failures don't crash the entire system
2. **Hot Restart**: Restart specific components without full system restart
3. **Remote Control**: Control system from remote machines via network
4. **Performance**: Multi-core utilization with separate processes
5. **Monitoring**: Centralized status monitoring and health checks
6. **Scalability**: Easy to add new GPIO controllers or sensors
7. **Testing**: Individual components can be tested in isolation