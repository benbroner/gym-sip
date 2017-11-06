import gym
from gym import error, spaces, utils
from gym.utils import seeding

import glob
import os
import random

import pandas as pd
import numpy as np

ACTION_SKIP = 0
ACTION_BUY = 1
ACTION_SELL = 2

class StockState:
    def __init__(self, equity_path, sep=','):
        df = self.read_csv(equity_path, sep=sep)

        df = df.fillna(method='ffill')
        df = df.fillna(method='bfill')

        self.df = df
        self.index = 0

        print("Imported tick data from {}".format(equity_path))

    def read_csv(self, path, sep):
        dtypes = {'Date': str, 'Time': str}
        df = pd.read_csv(path, sep=sep, header=0, names=['Date', 'Time', 'Open', 'High', 'Low', 'Close', 'Volume'], dtype=dtypes)
        dtime = df.Date + ' ' + df.Time
        df.index = pd.to_datetime(dtime)
        df.drop(['Date', 'Time'], axis=1, inplace=True)
        return df

    def reset(self):
        self.index = 0

    def next(self):
        if self.index >= len(self.df) - 1:
            return None, True

        values = self.df.iloc[self.index].values

        self.index += 1

        return values, False

    def shape(self):
        return self.df.shape

    def current_price(self):
        return self.df.ix[self.index, 'Close']

class StocksEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, datadir):
        self.bound = 100000

        self.comission = 0.1 / 100.
        self.num = 1

        self.money = 0
        self.equity = 0
        self.states = []
        self.state = None

        for path in glob.glob(datadir + '/*.csv'):
            if not os.path.isfile(path):
                continue

            self.states.append(path)

        self.observation_space = spaces.Box(low=0, high=self.bound, shape=(5,1))
        self.action_space = spaces.Discrete(3)

        if len(self.states) == 0:
            raise NameError('Invalid empty directory {}'.format(dirname))

    def _step(self, action):
        assert self.action_space.contains(action)

        portfolio = self.money + (1. - self.comission) * self.equity * self.state.current_price()
        price = self.state.current_price()
        cost = price * self.num
        comission_price = cost * (1. + self.comission)
        equity_price = price * self.equity
        prev_portfolio = self.money + equity_price

        if action == ACTION_BUY:
            if self.money >= comission_price:
                self.money -= comission_price
                self.equity += self.num
        if action == ACTION_SELL:
            if self.equity > 0:
                self.money += (1. - self.comission) * cost
                self.equity -= self.num

        state, done = self.state.next()

        new_price = price
        if not done:
            new_price = self.state.current_price()

        new_equity_price = new_price * self.equity
        reward = (self.money + new_equity_price) - prev_portfolio

        return state, reward, done, None

    def _reset(self):
        self.state = StockState(random.choice(self.states))

        self.money = 1000000
        self.equity = 0

        state, done = self.state.next()
        return state

    def _render(self, mode='human', close=False):
        pass
