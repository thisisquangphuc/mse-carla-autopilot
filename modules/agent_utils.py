# agent_utils.py

try:
    from leaderboard.autoagents.autonomous_agent import AutonomousAgent, Track  # type: ignore
except ImportError:
    class AutonomousAgent(object):
        pass

    class Track:
        SENSORS = 0
