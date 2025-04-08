# -------------------------
# Keyboard Control
# -------------------------
import sys
import carla
import pygame
from pygame.locals import K_DOWN, K_LEFT, K_RIGHT, K_SPACE, K_UP, K_a, K_d, K_s, K_w, K_q


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

