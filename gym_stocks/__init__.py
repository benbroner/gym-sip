from gym.envs.registration import register

register(
    id='Stocks-v0',
    entry_point='gym_stocks.envs:StocksEnv',
    kwargs={'datadir': 'stocks'}
)
