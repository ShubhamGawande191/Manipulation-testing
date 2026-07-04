"""Launch the agent-ros-bridge with SO101 arm plugin.

Usage:
    # Terminal 1: Launch ROS2 follower bringup
    source /opt/ros/jazzy/setup.bash && source ~/ros2_ws/install/setup.bash
    ros2 launch so101_bringup follower.launch.py use_rviz:=false use_cameras:=false

    # Terminal 2: Launch the bridge
    source /opt/ros/jazzy/setup.bash && source ~/ros2_ws/install/setup.bash
    cd /home/beelzebub/Desktop/RobotManipulator
    uv run python -m robot_manipulator.scripts.launch_bridge

    # Terminal 3: Test with wscat or the pick_and_place script
    wscat -c ws://localhost:8765
    > {"command": {"action": "arm.get_state"}}
    > {"command": {"action": "arm.home"}}
    > {"command": {"action": "arm.gripper", "parameters": {"position": 0.0}}}
"""

import asyncio
import logging
import os

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger("so101_bridge")


async def main():
    # JWT secret required by agent-ros-bridge
    if "JWT_SECRET" not in os.environ:
        os.environ["JWT_SECRET"] = os.urandom(32).hex()
        logger.info("Generated ephemeral JWT_SECRET (for local dev only)")

    from agent_ros_bridge import Bridge
    from agent_ros_bridge.gateway_v2.transports.websocket import WebSocketTransport

    from robot_manipulator.so101_bridge import SO101Plugin

    # Create bridge
    bridge = Bridge()

    # Register WebSocket transport
    ws_config = {"port": 8765, "auth": {"enabled": False}}
    bridge.transport_manager.register(WebSocketTransport(ws_config))

    # Create and register SO101 plugin
    plugin = SO101Plugin(namespace="follower")

    # Load plugin (connects to ROS2 and registers with bridge)
    await bridge.plugin_manager.load_plugin(plugin)

    logger.info("SO101 bridge running on ws://localhost:8765")
    logger.info(
        "Commands: arm.get_state, arm.home, arm.move_joints, arm.gripper, arm.stop, arm.camera"
    )

    # Patch: TransportManager._route_message is sync but WS handler awaits it.
    # Make it async so `await self.message_handler(msg, identity)` works properly.
    original_route = bridge.transport_manager._route_message

    async def _async_route(message, identity):
        if bridge.transport_manager._message_handler:
            return await bridge.transport_manager._message_handler(message, identity)

    for t in bridge.transport_manager.transports.values():
        t.message_handler = _async_route

    async with bridge.run():
        while bridge.running:
            await asyncio.sleep(1)


def app():
    asyncio.run(main())


if __name__ == "__main__":
    app()


def app():
    asyncio.run(main())


if __name__ == "__main__":
    app()
