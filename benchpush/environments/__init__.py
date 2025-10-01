from gymnasium.envs.registration import register

register(
     id="ship-ice-v0",
     entry_point="benchpush.environments.ship_ice_nav:ShipIceEnv",
     max_episode_steps=300,
)

register(
     id="ship-ice-mujoco-v0",
     entry_point="benchpush.environments.ship_ice_nav_mujoco:ShipIceMujoco",
     max_episode_steps=300,
)

register(
     id="box-delivery-v0",
     entry_point="benchpush.environments.box_delivery:BoxDeliveryEnv",
     max_episode_steps=30000,
)

register(
     id="box-delivery-mujoco-v0",
     entry_point="benchpush.environments.box_delivery_mujoco:BoxDeliveryMujoco",
     max_episode_steps=2000,
)

register(
     id="maze-NAMO-v0",
     entry_point="benchpush.environments.maze_NAMO:MazeNAMO",
     max_episode_steps=400,
)


register(
     id="maze-NAMO-mujoco-v0",
     entry_point="benchpush.environments.maze_NAMO_mujoco:MazeNAMOMujoco",
     max_episode_steps=2000,
)


register(
     id="area-clearing-mujoco-v0",
     entry_point="benchpush.environments.area_clearing_mujoco:AreaClearingMujoco",
     max_episode_steps=30000,
)

register(
     id="area-clearing-v0",
     entry_point="benchpush.environments.area_clearing:AreaClearingEnv",
     max_episode_steps=30000,
)