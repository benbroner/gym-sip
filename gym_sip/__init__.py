from gym.envs.registration import register

register(
    id='Sipper-v0',
    entry_point='gym_stocks.envs:SipEnv',
    kwargs={'file_name': 'data'}
)
