# CARLA Driver Assistant Agent - Lippstadt Summer School 2025

This project provides a unified environment for students to develop their driver assistant system using the CARLA simulator. It's designed specifically for the Lippstadt Summer School 2025, offering a framework for developing and testing various driver assistance features in a simulated environment.

## Project Overview

The Driver Assistant Agent (DAS) allows students to:
- Develop driver assistance algorithms in a controlled environment
- Test sensor fusion and environment perception systems
- Implement safety override mechanisms
- Evaluate their solutions in both standalone and evaluation modes

## Prerequisites

- Python 3.7
- CARLA Simulator (version 0.9.15)

## Project Structure

```
.
├── modules/           # Core DAS components
│   ├── sensors.py    # Sensor management (Camera, LIDAR, Radar, IMU)
│   ├── controls.py   # Vehicle control systems
│   ├── hud.py        # Heads-up display interface
│   └── agent_utils.py # Utility functions for the DAS agent
├── agent.py          # Main DAS agent implementation
├── local_mode.py     # Local testing mode
├── config.py         # DAS configuration settings
├── settings.ini      # Environment and simulation settings
├── setup_env.bat     # Environment setup script
└── requirements.txt  # Python dependencies
```

## DAS Components

The system includes several key components:

1. **Sensor Systems** (`modules/sensors.py`)
   - RGB Cameras (Center, Left, Right)
   - LIDAR sensor
   - Radar sensor
   - IMU (Inertial Measurement Unit)
   - Sensor fusion and data processing

2. **Control Systems** (`modules/controls.py`)
   - Vehicle control interface
   - Keyboard input handling
   - Control command processing

3. **User Interface** (`modules/hud.py`)
   - Heads-up display
   - Sensor data visualization
   - System status information

4. **Agent Logic** (`agent.py`)
   - Main decision loop
   - Human control processing
   - Assistant override implementation
   - Control merging logic

## Setup Instructions

1. **Install CARLA Simulator**
   - Download CARLA 0.9.15 from the official website
   - Extract it to a known location on your system

2. **Configure Settings**
   - Open `settings.ini` and update the following paths:
     - `CARLA_PATH_EGG`: Path to the CARLA egg file
     - `CARLA_PATH_WHL`: Path to the CARLA wheel file
   - Adjust sensor and system settings as needed:
     - Camera resolutions and FOV
     - LIDAR and Radar parameters
     - Control sensitivity

3. **Setup Python Environment**
   - Run the setup script:
     ```bash
     setup_env.bat
     ```
   This will:
   - Create a Python virtual environment
   - Install required dependencies
   - Install CARLA Python API
   - Configure the environment


## Running the DAS System
1. **Start CARLA Server**
   - Launch the CARLA simulator
   - Ensure it's running on the configured host and port (default: 127.0.0.1:2000)

2. **Run the DAS System**
   - For local testing:
     ```bash
     python local_mode.py
     ```
   - For full DAS execution:
     ```bash
     python agent.py
     ```

## Development Guidelines

### Implementing Driver Assistance Features

The main entry point for student development is the `get_assistant_override()` method in `agent.py`. This method should:

1. **Process Sensor Data**
   - Analyze camera images for image detection
   - Process LIDAR data for obstacle detection
   - Use radar data for object tracking
   - Incorporate IMU data for vehicle dynamics

2. **Implement Safety Logic**
   - Emergency braking for obstacles
   - Collision avoidance
   - Speed control in hazardous conditions

3. **Control Override Logic**
   - Determine when to override human control
   - Implement smooth control transitions
   - Ensure safety while respecting driver intent

### Example Implementation (pseudo code)

```python
def get_assistant_override(self, input_data):
    """
    Process sensor data and decide if an assistant override is needed.
    Students should implement override logic (e.g., emergency braking) here.
    """
    override_control = carla.VehicleControl()
    
    # Example: Emergency braking when obstacle detected
    if obstacle_detected(input_data.get('LIDAR')):
        override_control.brake = 1.0
        
    # Example: Lane departure warning
    if lane_departure_detected(input_data.get('Center')):
        # Apply corrective steering
        override_control.steer = calculate_corrective_steering()
        
    return override_control
```

## Testing Your Implementation

1. **Local Testing**
   - Use `local_mode.py` for quick iterations
   - Test basic functionality and sensor processing
   - Verify control override behavior

2. **Evaluation Mode**
   - The testing framework will inject sensor data and human control 
   - Your agent's API remains the same
   - Focus on implementing the `get_assistant_override()` method

## Troubleshooting

Common issues and solutions:

3. **Environment Setup Issues**
   - Make sure Python 3.7 is installed
   - Verify CARLA paths in `settings.ini`
   - Run `setup_env.bat` with administrator privileges if needed

## Contributing

1. Fork the repository
2. Create a team branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request
