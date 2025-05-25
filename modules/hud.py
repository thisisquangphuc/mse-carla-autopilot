
import numpy as np #type: ignore
import pygame #type: ignore
from os import environ
environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"

# -------------------------
# Display / HUD (Minimal)
# -------------------------


class HumanInterface(object):
    """
    Provides a display window for showing sensor images (and can serve as a minimal HUD).
    Used only in standalone mode.
    """
    def __init__(self, standalone, width, height, side_scale, left_mirror=True, right_mirror=True):
        self.standalone = standalone
        self._width = width
        self._height = height
        self._scale = side_scale
        self._left_mirror = left_mirror
        self._right_mirror = right_mirror
        self._font = pygame.font.Font("modules/assets/fonts/DS-DIGI.TTF", 30) #! Modified by Phuc - double check
        pygame.init()
        pygame.font.init()
        self._clock = pygame.time.Clock()
        self._display = pygame.display.set_mode((self._width, self._height),
                                                 pygame.HWSURFACE | pygame.DOUBLEBUF)
        pygame.display.set_caption("Driver Assistant Agent")
    
    def run_interface(self, input_data, pedestrian=False, lidar_point=[], velocity=0.0):
        """
        Run the GUI
        """
        # image_center = input_data['Center'][1]
        # self._surface = pygame.surfarray.make_surface(image_center.swapaxes(0, 1))
        # text_surface = self._font.render("Pedestrian Detected!", True, (255, 0, 0))
        # self._surface.blit(text_surface, (20, 20))

        if self.standalone:
            image_center = input_data['Center'][1]
            self._surface = pygame.surfarray.make_surface(image_center.swapaxes(0, 1))
            if self._left_mirror:
                image_left = input_data['Left'][1]
                left_surface = pygame.surfarray.make_surface(image_left.swapaxes(0, 1))
                self._surface.blit(left_surface, (0, (1 - self._scale) * self._height))
            if self._right_mirror:
                image_right = input_data['Right'][1]
                right_surface = pygame.surfarray.make_surface(image_right.swapaxes(0, 1))
                self._surface.blit(right_surface, ((1 - self._scale) * self._width, (1 - self._scale) * self._height))
        else:
            image_center = input_data['Center'][1][:, :, -2::-1]
            self._surface = pygame.surfarray.make_surface(image_center.swapaxes(0, 1))
            if self._left_mirror:
                image_left = input_data['Left'][1][:, :, -2::-1]
                left_surface = pygame.surfarray.make_surface(image_left.swapaxes(0, 1))
                self._surface.blit(left_surface, (0, (1 - self._scale) * self._height))
            if self._right_mirror:
                image_right = input_data['Right'][1][:, :, -2::-1]
                right_surface = pygame.surfarray.make_surface(image_right.swapaxes(0, 1))
                self._surface.blit(right_surface, ((1 - self._scale) * self._width, (1 - self._scale) * self._height))

        self.draw_alert(f"IMU-based speed estimate: {velocity:.2f} m/s", (255,0,0), (self._width/2, self._height/2))
        if pedestrian: 
            self.draw_alert("Pedestrian Detected!", (255, 0, 0), (self._width/3, self._height-50))

        # if lidar_point is not None:
        #     for point in lidar_point:
        #         self.draw_point(np.array(point))
        #         # print(np.array(point))

        if self._surface is not None:
            self._display.blit(self._surface, (0, 0))
        pygame.display.flip()
    
    def set_black_screen(self):
        """
        Clear the display to a black screen.
        """
        black = np.zeros((self._height, self._width, 3), dtype=np.uint8)
        surface = pygame.surfarray.make_surface(black.swapaxes(0, 1))
        self._display.blit(surface, (0, 0))
        pygame.display.flip()

    def draw_alert(self, message, color=(255, 0, 0), pos=(0,0)):
        text_surface = self._font.render(message, True, color)
        self._surface.blit(text_surface, pos)
        # get screen solution and scale it in the middle
        # Detect signs and take control (speed)
        # Traffic light detection

    def draw_point(self, coordinate, color=(255,0,0)):
        text_surface = self._font.render("xx", True, color)
        self._surface.blit(text_surface, (coordinate[0]+1, coordinate[1]+1))

    def _quit(self):
        pygame.quit()


