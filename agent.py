#!/usr/bin/env python

"""
Driver Assistant Agent for CARLA - Lippstadt Summer School 2025

This agent serves as the base for developing a Driver Assistant System (DAS).
Students will primarily modify the `get_assistant_override` method to implement
features like emergency braking, camera assist, etc.

Modes of Operation:
- Standalone Mode: Runs independently using `local_mode.py`. Spawns its own
                   vehicle, sensors, and uses keyboard input for human control.
                   Useful for development and testing.

- Evaluation Mode: Integrated into a testing framework. Sensor data and human
                   control are injected externally. The agent's core logic
                   (`run_step`) remains the same.
"""

# -------------------------
# Imports
# -------------------------

import carla #type: ignore
from modules.controls import KeyboardControl
from modules.hud import HumanInterface
from modules.agent_utils import AutonomousAgent, Track
from modules.sensors import CameraManager, IMUSensor, LidarManager, RadarSensor
from modules.sensors import get_sensor_layout
import logging
import contextlib
with contextlib.redirect_stdout(None):
    import pygame

# -------------------------
# Entry Point (DO NOT ALTER!)
# -------------------------
def get_entry_point():
    """Returns the main agent class name."""
    return 'DriverAssistantAgent'
# -------------------------
# Main Agent Class
# -------------------------
class DriverAssistantAgent(AutonomousAgent):
    """
    The main agent class. It processes sensor data, handles human input (keyboard/steering wheel),
    allows for assistant overrides, and outputs final vehicle control commands.
    """
    def setup(self, path_to_conf_file, standalone_mode=False):
        """
        Initialize the agent.
        
        Args:
            path_to_conf_file: Path to configuration file (if needed for keyboard control).
            standalone_mode: If True, the agent spawns its own sensors.
                             If False, sensor and control data are provided externally by the evaluation framework.
        """
        self.track = Track.SENSORS  
        self.agent_engaged = False
        self.standalone_mode = standalone_mode
        self.camera_width = 1280
        self.camera_height = 720
        self._side_scale = 0.3
        self._left_mirror = True
        self._right_mirror = True
        self._prev_timestamp = 0
        self._clock = pygame.time.Clock()
        self._hic = HumanInterface(self.standalone_mode, self.camera_width, self.camera_height, self._side_scale, self._left_mirror, self._right_mirror)
        self._controller = KeyboardControl(path_to_conf_file)
        
        #----------------------------------------------------------
        # Initialize any other variables needed for your agent here
        # --- Agent Setup (HERE) ---
        #----------------------------------------------------------
        # (CODE TO BE ADDED BY STUDENTS) [Optional]
        # ----------------------------------------------------------



        # --- Sensor Spawning (Local Mode Only) ---
        if self.standalone_mode:
            self._sensor_objects = {}
            self._spawn_sensors()
        logging.info("DriverAssistantAgent setup complete. Standalone mode: %s", self.standalone_mode)

    def sensors(self):
        """
        Define the sensor suite required by the agent.
        """
        sensors = get_sensor_layout(self.camera_width, self.camera_height, self._side_scale)
        return sensors

    def _spawn_sensors(self):
        """
        Spawns sensors based on the definitions in sensors().
        Their callbacks store the actual sensor outputs.
        Used only in standalone mode.
        """
        world = self.vehicle.get_world()
        bp_library = world.get_blueprint_library()
        for sensor_def in self.sensors():
            sensor_id = sensor_def['id']
            bp = bp_library.find(sensor_def['type'])
            if sensor_def['type'].startswith('sensor.camera'):
                bp.set_attribute('image_size_x', str(sensor_def['width']))
                bp.set_attribute('image_size_y', str(sensor_def['height']))
                bp.set_attribute('fov', str(sensor_def['fov']))
            elif sensor_def['type'] == 'sensor.other.radar':
                bp.set_attribute('horizontal_fov', str(sensor_def.get('horizontal_fov', 30.0)))
                bp.set_attribute('vertical_fov', str(sensor_def.get('vertical_fov', 30.0)))

            transform = carla.Transform(
                carla.Location(x=sensor_def['x'], y=sensor_def.get('y', 0.0), z=sensor_def['z']),
                carla.Rotation(
                    roll=sensor_def.get('roll', 0.0),
                    pitch=sensor_def.get('pitch', 0.0),
                    yaw=sensor_def.get('yaw', 0.0)
                )
            )
            sensor_actor = world.spawn_actor(bp, transform, attach_to=self.vehicle)
            if sensor_def['type'].startswith('sensor.camera'):
                cam_manager = CameraManager(self.vehicle, self._hic)
                cam_manager.set_sensor(sensor_actor)
                self._sensor_objects[sensor_id] = cam_manager
            elif sensor_def['type'].startswith('sensor.lidar'):
                lidar_manager = LidarManager(self.vehicle, sensor_def)
                lidar_manager.set_sensor(sensor_actor)
                self._sensor_objects[sensor_id] = lidar_manager
            elif sensor_def['type'] == 'sensor.other.radar':
                radar_sensor = RadarSensor(self.vehicle)
                self._sensor_objects[sensor_id] = radar_sensor
            elif sensor_def['type'] == 'sensor.other.imu':
                imu_sensor = IMUSensor(self.vehicle)
                self._sensor_objects[sensor_id] = imu_sensor

    def get_sensor_data(self):
        """
        Retrieves the latest data from all managed sensors.
        This is called internally in `run_step` when in standalone mode.
        In evaluation mode, the framework provides this data.

        Returns:
            A dictionary where keys are sensor IDs and values are tuples:
            (sensor_actor, sensor_data). The sensor_actor might be None
            if the manager processes the data directly.
            Example: {'Center': (None, pygame_surface), 'LIDAR': (None, point_cloud)}
        """
        sensor_data = {}
        for sensor_id, sensor_obj in self._sensor_objects.items():
            if sensor_id in ['Center', 'Left', 'Right']:
                if sensor_obj.latest_image is not None:
                    sensor_data[sensor_id] = (None, sensor_obj.latest_image)
            elif sensor_id == 'LIDAR':
                if sensor_obj.latest_data is not None:
                    sensor_data[sensor_id] = (None, sensor_obj.latest_data)
        return sensor_data

    def get_human_control(self, timestamp):
        """
        Retrieve human control commands.
        """
        time_diff = timestamp - self._prev_timestamp
        return self._controller.parse_events(time_diff)

    # ==============================================================================
    # -- STUDENT IMPLEMENTATION AREA START -----------------------------------------
    # ==============================================================================

    def get_assistant_override(self, input_data):
        """
        *** THIS IS THE MAIN FUNCTION STUDENTS NEED TO MODIFY ***

        Process sensor data and decide if the Driver Assistant System (DAS)
        should override the human driver's input.

        Args:
            input_data (dict): A dictionary containing the latest sensor data,
                               structured as returned by `get_sensor_data`.
                               Example keys: 'Center', 'Left', 'Right' (pygame surfaces),
                               'LIDAR' (carla.LidarMeasurement), 'RADAR', 'IMU', etc.
                               Access data like: `image = input_data.get('Center')[1]`
                               or `lidar_data = input_data.get('LIDAR')[1]`
                               Check if a key exists before accessing it!

        Returns:
            A carla.VehicleControl object.
            - Set fields (e.g., `brake=1.0`) to override the corresponding
              human input.
            - Leave fields at their default values (0.0 for steer, throttle, brake;
              False for hand_brake, reverse) if no override is intended for that
              specific control.
            - The `merge_control` function will prioritize non-default values
              from this object over the human input.
        """
        # --- Student Implementation Example ---
        override_control = carla.VehicleControl() # Start with a default (no override) control object

        # --- Example 1: Simple Emergency Braking based on LIDAR ---
        lidar_measurement = input_data.get('LIDAR')
        if lidar_measurement:
            lidar_data = lidar_measurement[1] # Get the actual data
            min_distance = float('inf')
            # Process lidar points (this is a basic example, needs refinement)
            for location in lidar_data:
                # Calculate distance (ignoring z for simplicity here)
                distance = (location.x**2 + location.y**2)**0.5
                # Check if the point is roughly in front
                if distance < min_distance and abs(location.y) < 1.0 and location.x > 0:
                     min_distance = distance

            # If an obstacle is detected very close in front
            if min_distance < 5.0: # Threshold distance in meters
                logging.warning("DAS: Obstacle detected close ahead (%.2f m)! Applying brake override.", min_distance)
                override_control.brake = 1.0  # Override: Full brake
                override_control.throttle = 0.0 # Override: Ensure no throttle
        
        
        return override_control
    # ==============================================================================
    # -- STUDENT IMPLEMENTATION AREA END -------------------------------------------
    # ==============================================================================

    def merge_control(self, human_control, assistant_override):
        """
        Merges the human control input with the assistant's override commands.
        The assistant override takes precedence for steering, throttle, and brake
        if the override value is non-zero (or specifically set).

        Args:
            human_control: carla.VehicleControl from human input.
            assistant_override: carla.VehicleControl from `get_assistant_override`.

        Returns:
            The final carla.VehicleControl to be applied to the vehicle.
        """
        final_control = carla.VehicleControl()
        final_control.steer = assistant_override.steer if abs(assistant_override.steer) > 0.0 else human_control.steer
        final_control.throttle = (assistant_override.throttle 
                                  if assistant_override.throttle > human_control.throttle 
                                  else human_control.throttle)
        final_control.brake = max(human_control.brake, assistant_override.brake)
        final_control.hand_brake = human_control.hand_brake or assistant_override.hand_brake
        final_control.gear = human_control.gear
        final_control.reverse = human_control.reverse
        return final_control

    def run_step(self, input_data=None, timestamp=0):
        """
        Main decision loop: obtain sensor data (if in standalone mode), read human inputs,
        compute any assistant overrides, merge both, and return the final control command.
        This method conforms to the Leaderboard agent API.
        """
        self._clock.tick_busy_loop(120)
        self.agent_engaged = True
        pygame.event.pump()
        if self.standalone_mode:
            if input_data is None:
                input_data = self.get_sensor_data()
        self._hic.run_interface(input_data)
        human_control = self.get_human_control(timestamp)
        assistant_override = self.get_assistant_override(input_data)
        final_control = self.merge_control(human_control, assistant_override)
        logging.info("Timestamp: %f", timestamp)
        logging.info("Human Control: %s", human_control)
        logging.info("Assistant Override: %s", assistant_override)
        logging.info("Final Merged Control: %s", final_control)
        self._prev_timestamp = timestamp
        return final_control

    def destroy(self):
        """
        Cleanup: destroy spawned sensors and close the HUD (if in standalone mode).
        """
        if self.standalone_mode:
            for sensor_obj in self._sensor_objects.values():
                if sensor_obj.sensor is not None:
                    sensor_obj.sensor.destroy()
            self._hic.set_black_screen()
            self._hic._quit()


