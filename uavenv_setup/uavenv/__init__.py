from gym.envs.registration import register

register(
    id='uavenv-v0',
    entry_point='uavenv.envs:UavEnv',
)