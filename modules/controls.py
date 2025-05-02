import sys
import carla #type: ignore
import pygame #type: ignore
from configparser import ConfigParser
from pygame.locals import K_DOWN, K_LEFT, K_RIGHT, K_SPACE, K_UP, K_a, K_d, K_s, K_w, K_q #type: ignore

class KeyboardControl(object):
    """
    Reads keyboard events and converts them into basic vehicle control commands.
    Also toggles reverse mode when the Q key is pressed.
    Active in standalone mode.
    """
    def __init__(self, config_file=None):
        self._control = carla.VehicleControl()
        self._steer_cache = 0.0
        self._clock = pygame.time.Clock()
        self._mode = "normal"  # For simplicity, we assume normal mode.

    def parse_events(self, time_diff):
        # Process events for quitting and reverse toggle.
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYUP:
                if event.key == K_q:
                    self._control.gear = 1 if self._control.reverse else -1
                    self._control.reverse = self._control.gear < 0

        keys = pygame.key.get_pressed()
        # Throttle: Up arrow or 'W'
        self._control.throttle = 0.6 if (keys[K_UP] or keys[K_w]) else 0.0
        # Steer: Left/Right arrows or A/D keys
        steer_increment = 3e-4 * time_diff * 1000
        if keys[K_LEFT] or keys[K_a]:
            self._steer_cache -= steer_increment
        elif keys[K_RIGHT] or keys[K_d]:
            self._steer_cache += steer_increment
        else:
            self._steer_cache = 0.0
        self._control.steer = round(self._steer_cache, 1)
        # Brake and hand brake.
        self._control.brake = 1.0 if (keys[K_DOWN] or keys[K_s]) else 0.0
        self._control.hand_brake = keys[K_SPACE]
        return self._control

class DualControl(object):
    def __init__(self, config_file=None):

        self._steer_cache = 0.0
        pygame.joystick.init()

        joystick_count = pygame.joystick.get_count()
        if joystick_count > 1:
            raise ValueError("Please Connect Just One Joystick")

        self._joystick = pygame.joystick.Joystick(0)
        self._joystick.init()

        self._parser = ConfigParser()
        self._parser.read('wheel_config.ini')
        self._steer_idx = int(
            self._parser.get('G29 Racing Wheel', 'steering_wheel'))
        self._throttle_idx = int(
            self._parser.get('G29 Racing Wheel', 'throttle'))
        self._brake_idx = int(self._parser.get('G29 Racing Wheel', 'brake'))
        self._reverse_idx = int(self._parser.get('G29 Racing Wheel', 'reverse'))
        self._handbrake_idx = int(
            self._parser.get('G29 Racing Wheel', 'handbrake'))

        self._control = carla.VehicleControl()

    def parse_events(self, time_diff):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.JOYBUTTONDOWN:
                if event.button == self._reverse_idx:
                    self._control.gear = 1 if self._control.reverse else -1
                    self._control.reverse = self._control.gear < 0

        self._parse_vehicle_wheel(time_diff)
        return self._control

    def _parse_vehicle_wheel(self, time_diff):
        numAxes = self._joystick.get_numaxes()
        jsInputs = [float(self._joystick.get_axis(i)) for i in range(numAxes)]
        jsButtons = [float(self._joystick.get_button(i)) for i in
                     range(self._joystick.get_numbuttons())]

        # Steering
        steerCmd = jsInputs[self._steer_idx]
        steer_increment = 3e-4 * time_diff * 1000
        self._steer_cache = steerCmd
        self._control.steer = round(self._steer_cache, 1)

        # Throttle
        throttleCmd = (1 - jsInputs[self._throttle_idx]) / 2.0
        self._control.throttle = throttleCmd

        # Brake
        brakeCmd = (1 - jsInputs[self._brake_idx]) / 2.0
        self._control.brake = brakeCmd

        # Handbrake
        self._control.hand_brake = bool(jsButtons[self._handbrake_idx])