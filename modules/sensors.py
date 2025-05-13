# -------------------------
# Sensor Managers
# -------------------------
import carla #type: ignore
from   carla import ColorConverter as cc #type: ignore
import numpy as np #type: ignore
import logging
import weakref
import math
logger = logging.getLogger(__name__)


# -------------------------
# Sensor layout
# -------------------------

def get_sensor_layout(camera_width, camera_height, side_scale):
    
    sensors = [
            {'type': 'sensor.camera.rgb', 'x': 0.7, 'y': 0.0, 'z': 1.60, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
             'width': camera_width, 'height': camera_height, 'fov': 100, 'id': 'Center'},
            {'type': 'sensor.camera.rgb', 'x': 0.7, 'y': -1, 'z': 1.0, 'roll': 0.0, 'pitch': 0.0, 'yaw': -135.0,
             'width': int(camera_width * side_scale), 'height': int(camera_height * side_scale), 'fov': 100, 'id': 'Left'},
            {'type': 'sensor.camera.rgb', 'x': 0.7, 'y': 1, 'z': 1.0, 'roll': 0.0, 'pitch': 0.0, 'yaw': 135.0,
             'width': int(camera_width * side_scale), 'height': int(camera_height * side_scale), 'fov': 100, 'id': 'Right'},
            {'type': 'sensor.lidar.ray_cast', 'x': 0.7, 'y': -0.4, 'z': 1.60, 'roll': 0.0, 'pitch': 0.0,
             'yaw': -45.0, 'id': 'LIDAR'},
            {'type': 'sensor.other.radar', 'x': 0.7, 'y': -0.4, 'z': 1.60, 'roll': 0.0, 'pitch': 0.0,
             'yaw': -45.0, 'fov': 30, 'id': 'RADAR', 'horizontal_fov' : 30.0, 'vertical_fov' : 30.0},
            {'type': 'sensor.other.imu', 'x': 0.7, 'y': -0.4, 'z': 1.60, 'roll': 0.0, 'pitch': 0.0,
             'yaw': -45.0, 'id': 'IMU'},        ]
        
    return sensors

# -------------------------
# Local Sensor Managers
# -------------------------

class CameraManager(object):
    """
    Manages a camera sensor and stores its latest image.
    """
    def __init__(self, parent_actor, hud):
        self.sensor = None
        self.latest_image = None
        self._parent = parent_actor
        self.hud = hud
        self.recording = False
        self.sensors = [['sensor.camera.rgb', cc.Raw, 'Camera RGB']]
        self.index = 0

    def set_sensor(self, sensor_actor):
        """Register the spawned sensor actor and its callback."""
        self.sensor = sensor_actor
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda image: CameraManager._parse_image(weak_self, image))

    def set_recording(self, on=True):
        self.recording = on

    @staticmethod
    def _parse_image(weak_self, image):
        self = weak_self()
        if not self:
            return
        try:
            image.convert(self.sensors[self.index][1])
            array = np.frombuffer(image.raw_data, dtype=np.uint8)
            array = np.reshape(array, (image.height, image.width, 4))
            # Remove alpha channel and convert from BGR to RGB.
            array = array[:, :, :3][:, :, ::-1]
            self.latest_image = array
            if self.recording:
                image.save_to_disk('no_pedestrian/%08d.jpg' % image.frame) # Save images to disk (Careful with large datasets and capute sequence!)
                self.recording = False

        except Exception as e:
            logging.warning("Error parsing camera image: %s", e)

class LidarManager(object):
    """
    Manages a LIDAR sensor and stores its latest point cloud.
    """
    def __init__(self, parent_actor, sensor_def):
        self.sensor = None
        self.latest_data = None
        self._parent = parent_actor
        self.sensor_def = sensor_def

    def set_sensor(self, sensor_actor):
        self.sensor = sensor_actor
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda data: LidarManager._on_lidar(weak_self, data))

    @staticmethod
    def _on_lidar(weak_self, data):
        self = weak_self()
        if not self:
            return
        try:
            points = np.frombuffer(data.raw_data, dtype=np.float32)
            points = np.reshape(points, (int(points.shape[0] / 4), 4))
            self.latest_data = points
        except Exception as e:
            logging.warning("Error processing lidar data: %s", e)

class RadarSensor(object):
    def __init__(self, parent_actor):
        self.sensor = None
        self._parent = parent_actor
        bound_x = 0.5 + self._parent.bounding_box.extent.x
        bound_y = 0.5 + self._parent.bounding_box.extent.y
        bound_z = 0.5 + self._parent.bounding_box.extent.z

        self.velocity_range = 7.5 # m/s
        world = self._parent.get_world()
        self.debug = world.debug
        bp = world.get_blueprint_library().find('sensor.other.radar')
        bp.set_attribute('horizontal_fov', str(35))
        bp.set_attribute('vertical_fov', str(20))
        self.sensor = world.spawn_actor(
            bp,
            carla.Transform(
                carla.Location(x=bound_x + 0.05, z=bound_z+0.05),
                carla.Rotation(pitch=5)),
            attach_to=self._parent)
        # We need a weak reference to self to avoid circular reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(
            lambda radar_data: RadarSensor._Radar_callback(weak_self, radar_data))

    @staticmethod
    def _Radar_callback(weak_self, radar_data):
        self = weak_self()
        if not self:
            return
        # To get a numpy [[vel, altitude, azimuth, depth],...[,,,]]:
        # points = np.frombuffer(radar_data.raw_data, dtype=np.dtype('f4'))
        # points = np.reshape(points, (len(radar_data), 4))

        current_rot = radar_data.transform.rotation
        for detect in radar_data:
            azi = math.degrees(detect.azimuth)
            alt = math.degrees(detect.altitude)
            # The 0.25 adjusts a bit the distance so the dots can
            # be properly seen
            fw_vec = carla.Vector3D(x=detect.depth - 0.25)
            carla.Transform(
                carla.Location(),
                carla.Rotation(
                    pitch=current_rot.pitch + alt,
                    yaw=current_rot.yaw + azi,
                    roll=current_rot.roll)).transform(fw_vec)

            def clamp(min_v, max_v, value):
                return max(min_v, min(value, max_v))

            norm_velocity = detect.velocity / self.velocity_range # range [-1, 1]
            r = int(clamp(0.0, 1.0, 1.0 - norm_velocity) * 255.0)
            g = int(clamp(0.0, 1.0, 1.0 - abs(norm_velocity)) * 255.0)
            b = int(abs(clamp(- 1.0, 0.0, - 1.0 - norm_velocity)) * 255.0)
            # self.debug.draw_point(
            #     radar_data.transform.location + fw_vec,
            #     size=0.075,
            #     life_time=0.06,
            #     persistent_lines=False,
            #     color=carla.Color(r, g, b))

class IMUSensor(object):
    def __init__(self, parent_actor):
        self.sensor = None
        self._parent = parent_actor
        self.accelerometer = (0.0, 0.0, 0.0)
        self.gyroscope = (0.0, 0.0, 0.0)
        self.compass = 0.0
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.imu')
        self.sensor = world.spawn_actor(
            bp, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(
            lambda sensor_data: IMUSensor._IMU_callback(weak_self, sensor_data))

    @staticmethod
    def _IMU_callback(weak_self, sensor_data):
        self = weak_self()
        if not self:
            return
        limits = (-99.9, 99.9)
        self.accelerometer = (
            max(limits[0], min(limits[1], sensor_data.accelerometer.x)),
            max(limits[0], min(limits[1], sensor_data.accelerometer.y)),
            max(limits[0], min(limits[1], sensor_data.accelerometer.z)))
        self.gyroscope = (
            max(limits[0], min(limits[1], math.degrees(sensor_data.gyroscope.x))),
            max(limits[0], min(limits[1], math.degrees(sensor_data.gyroscope.y))),
            max(limits[0], min(limits[1], math.degrees(sensor_data.gyroscope.z))))
        self.compass = math.degrees(sensor_data.compass)
