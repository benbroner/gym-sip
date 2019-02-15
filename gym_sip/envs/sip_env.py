import gym
from gym import spaces
from sklearn.preprocessing import RobustScaler
import sippy_lines as sl
import random
import pandas as pd
import numpy as np

ACTION_SKIP = 0
ACTION_BUY_A = 1
ACTION_BUY_H = 2
# ACTION_UPDATE_LIVE = 3
# ACTION_UPDATE_PRE = 4
# ACTION_GAMESCORE = 5#


class SippyState:
    def __init__(self, file_name):
        self.fn = file_name + '.csv'
        self.df = None
        self.read_csv()
        self.index = 0

        print("Imported data from {}".format(self.fn))

    def read_csv(self):
        df = pd.read_csv(self.fn)
        self.df = df.dropna()

    def fit_data(self):
        transformer = RobustScaler().fit(self.df)
        transformer.transform(self.df)

    def reset(self):
        self.index = 0

    def next(self):
        if self.index >= len(self.df) - 1:
            return None, True

        values = self.df.iloc[self.index, 5:30].values

        self.index += 1

        return values, False

    def shape(self):
        return self.df.shape

    def current_price(self):
        return self.df.ix[self.index, 'Close']


class SipEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, file_name):
        self.file_name = file_name
        self.num = 1
        self.money = 3500
        self.eq_a = 0
        self.eq_h = 0
        self.states = []
        self.state = None
        self.states.append(self.file_name)

        self.observation_space = spaces.Box(low=-, high=self.bound, shape=(5,1))
        self.action_space = spaces.Discrete(3)

        if len(self.states) == 0:
            raise NameError('Invalid empty directory {}'.format(self.file_name))

    def _step(self, action):
        assert self.action_space.contains(action)

        portfolio = self.money * self.equity * self.state.current_price()
        price = self.state.current_price()
        cost = price * self.num
        equity_price = price * self.equity
        prev_portfolio = self.money + equity_price

        if action == ACTION_BUY_A:
            if self.money >= 100 * self.num:
                self.money -= 100 * self.num
                self.eq_a += self.num
        if action == ACTION_BUY_H:
            if self.money >= 100 * self.num:
                self.money -= 100 * self.num
                self.eq_h += self.num

        state, done = self.state.next()

        new_price = price
        if not done:
            new_price = self.state.current_price()

        new_equity_price = new_price * self.equity
        reward = (self.money + new_equity_price) - prev_portfolio

        return state, reward, done, None

    def _reset(self):
        self.state = SippyState(random.choice(self.states))

        self.money = 3500
        self.equity = 0

        state, done = self.state.next()
        return state

    def _render(self, mode='human', close=False):
        pass
