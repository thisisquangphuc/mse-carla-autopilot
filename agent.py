#!/usr/bin/env python

"""
Driver Assistant Agent for CARLA - Lippstadt Summer School 2025

This script provides a unified environment for students to develop their driver assistant system.
In standalone mode, the script loads configuration settings, creates a CARLA client/world, spawns a vehicle,
spawns sensors, creates a HUD, and reads keyboard input.

In evaluation mode, the testing framework injects sensor data and human control, while the agent’s API remains the same.
Students can implement their assistant override logic in get_assistant_override() without worrying about the underlying integration.
"""

import carla
from modules.controls import KeyboardControl
from modules.hud import HumanInterface
from modules.agent_utils import AutonomousAgent, Track
from modules.sensors import CameraManager, IMUSensor, LidarManager, RadarSensor
from modules.sensors import get_sensor_layout
import logging
import contextlib
with contextlib.redirect_stdout(None):
    import pygame
import tensorflow as tf
import keras
import numpy as np
from PIL import Image
# from tensorflow.keras.preprocessing import image
logger = logging.getLogger(__name__)

# -------------------------
# Entry Point (DO NOT ALTER!)
# -------------------------
def get_entry_point():
    return 'DriverAssistantAgent'

# -------------------------
# Main Agent Class
# -------------------------
class DriverAssistantAgent(AutonomousAgent):
    """
    Main agent class that processes sensor data and generates vehicle control commands.
    In standalone mode the agent spawns its own sensors; in evaluation mode these are provided externally.
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
        self.camera_width = 800
        self.camera_height = 600
        self._side_scale = 0.3
        self._left_mirror = True
        self._right_mirror = True
        self._prev_timestamp = 0
        self._clock = pygame.time.Clock()
        self._hic = HumanInterface(self.standalone_mode, self.camera_width, self.camera_height, self._side_scale, self._left_mirror, self._right_mirror)
        self._controller = KeyboardControl(path_to_conf_file)
        if self.standalone_mode:
            self._sensor_objects = {}
            self._spawn_sensors()
        self.model = keras.models.load_model('model/pedestrian_model.keras')
        self.pre_status=0
        self.cur_status=0
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
        Gather the latest sensor outputs into a dictionary.
        In standalone mode, this queries the spawned sensors.
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

    def get_front_image_frame(self):
        """
        """
        sensor_obj = self._sensor_objects['Center']
        if sensor_obj.latest_image is not None:
            sensor_obj.set_recording()
        
    def get_left_image_frame(self):
        """
        """
        sensor_obj = self._sensor_objects['Left']
        if sensor_obj.latest_image is not None:
            sensor_obj.set_recording()
        
    def get_right_image_frame(self):
        """
        """    
        sensor_obj = self._sensor_objects['Right']
        if sensor_obj.latest_image is not None:
            sensor_obj.set_recording()

    def get_human_control(self, input_data, timestamp):
        """
        Retrieve human control commands.
        """
        time_diff = timestamp - self._prev_timestamp
        return self._controller.parse_events(time_diff)

    def get_assistant_override(self, input_data):
        """
        Process sensor data and decide if an assistant override is needed.
        Students should implement override logic (e.g., emergency braking) here.
        By default, no override is applied.
        """
        override_control = carla.VehicleControl()
        # Example placeholder logic:
        # if obstacle_detected(input_data.get('LIDAR')):
        #     override_control.brake = 1.0
        sensor_obj = self._sensor_objects['Center']
        if sensor_obj.latest_image is not None:
            image = Image.fromarray(sensor_obj.latest_image)
            image = image.resize((224, 224))
            array = np.array(image) / 255.0 # Normalize to [0, 1]
            # array = cv2.resize(sensor_obj.latest_image, (224, 224), interpolation=cv2.INTER_AREA)
            # array = array / 255.0  # Normalize to [0, 1]
            # Add batch dimension
            input_tensor = np.expand_dims(array, axis=0)  # Now (1, 224, 224, 3)
            prediction = self.model.predict(input_tensor)
            if prediction[0][0] > 0.5:
            #     self.cur_status=1
            # else:
            #     self.cur_status=0

            # if self.cur_status != self.pre_status:
            #     self.pre_status = self.cur_status 
                # self._hic.run_interface(input_data, True)
                self._hic.run_interface_w_alert(input_data)
            # else:
                # self._hic.run_interface(input_data, False)

        return override_control

    def merge_control(self, human_control, assistant_override):
        """
        Merge human input with any assistant override commands.
        Override commands (if non-zero) take priority over human input.
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

        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN and event.key == pygame.K_p:
                self.get_front_image_frame()
                break

        if self.standalone_mode:
            if input_data is None:
                input_data = self.get_sensor_data()
        self._hic.run_interface(input_data)
        # self._hic.run_interface_w_alert(input_data)
        human_control = self.get_human_control(input_data, timestamp)
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


