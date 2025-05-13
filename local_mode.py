# -------------------------
# Configuration Utilities
# -------------------------
import configparser
import logging
import carla #type: ignore
import contextlib
with contextlib.redirect_stdout(None):
    import pygame #type: ignore
from agent import DriverAssistantAgent

logger = logging.getLogger(__name__)

def load_config(config_file='settings.ini'):
    """
    Load configuration settings from a .ini file.
    Expected sections: CARLA-SERVER and CARLA-CLIENT.
    """
    config = configparser.ConfigParser()
    config.read(config_file)
    class Args:
        pass
    args = Args()
    # CARLA server settings
    server = config['CARLA-SERVER']
    args.host = server.get('host', '127.0.0.1')
    args.port = server.getint('port', 2000)
    # CARLA client settings
    client_conf = config['CARLA-CLIENT']
    res = client_conf.get('resolution', '800x600')
    args.width, args.height = [int(x) for x in res.split('x')]
    args.filter = client_conf.get('filter', 'vehicle.*')
    args.autopilot = client_conf.getboolean('autopilot', False)
    args.debug = client_conf.getboolean('debug', True)
    return args

# -------------------------
# Main Execution Loop (Local Mode)
# -------------------------
def main():
    args = load_config('settings.ini')
    pygame.init()
    pygame.font.init()

    logging.basicConfig(filename='run_log/run.log', level=logging.WARN)
    logger.info('Started')

    # Create CARLA client and obtain the world.
    client = carla.Client(args.host, args.port)
    client.set_timeout(20.0)
    world = client.get_world()
    blueprint_library = world.get_blueprint_library()

    vehicle_bp = blueprint_library.find("vehicle.lincoln.mkz_2017")
    if vehicle_bp is None:
        logging.error("vehicle.lincoln.mkz_2017 not found. Falling back to default filter.")
        vehicle_bp = blueprint_library.filter(args.filter)[0]

    spawn_points = world.get_map().get_spawn_points()
    spawn_point = spawn_points[0] if spawn_points else carla.Transform()
    vehicle = world.try_spawn_actor(vehicle_bp, spawn_point)
    if vehicle is None:
        logging.error("Failed to spawn vehicle.")
        return
    
    agent = DriverAssistantAgent()
    agent.vehicle = vehicle
    print("Running in standalone mode")
    agent.setup(None, standalone_mode=True)
    
    clock = pygame.time.Clock()
    num = 0

    # get_pedestrian = False
    # pedestrian = []

    try:
        while True:
            clock.tick_busy_loop(120)
            # if not get_pedestrian:
            #     if len(world.get_actors().filter('vehicle.*') > 0):
            #         get_pedestrian = True
            #         pedestrian.append(world.get_actors().filter('walker.*'))
# 
            if (pygame.time.get_ticks() > (num*5000)):
                # agent.get_front_image_frame()
                # if ((num % 3) == 0):
                #     agent.get_left_image_frame()
                # if ((num % 4) == 0):
                #     agent.get_right_image_frame()
                num += 1
            timestamp = world.get_snapshot().timestamp.elapsed_seconds
            control = agent.run_step(timestamp=timestamp)
            vehicle.apply_control(control)
            pygame.display.flip()
    except KeyboardInterrupt:
        pass
    finally:
        agent.destroy()
        vehicle.destroy()
        pygame.quit()
        logging.info("Shutdown complete.")

if __name__ == '__main__':
    main()