import gym
import random
import pandas as pd
import numpy as np
from gym import spaces
from sklearn.preprocessing import RobustScaler

ACTION_SKIP = 0
ACTION_BUY_A = 1
ACTION_BUY_H = 2


class SippyState:
    def __init__(self, file_name):
        self.fn = file_name + '.csv'
        self.df = None
        self.headers = ['a_team', 'h_team', 'league', 'game_id',
                        'a_pts', 'h_pts', 'secs', 'status', 'a_win', 'h_win', 'last_mod_to_start',
                        'num_markets', 'a_odds_ml', 'h_odds_ml', 'a_hcap_tot', 'h_hcap_tot']
        self.dtypes = {
                    'a_team': 'category',
                    'h_team': 'category',
                    'league': 'category',
                    'game_id': 'Int32',
                    'a_pts': 'Int32',
                    'h_pts': 'Int32',
                    'secs': 'Int32',
                    'status': 'Int32',
                    'a_win': 'Int32',
                    'h_win': 'Int32',
                    'num_markets': 'Int32',
                    'a_odds_ml': 'Int32',
                    'h_odds_ml': 'Int32',
                    'a_hcap_tot': 'Int32',
                    'h_hcap_tot': 'Int32'
        }
        self.read_csv()
        self.shape()

        self.chunk_df()
        # self.fit_data()
        self.index = 0

        print("Imported data from {}".format(self.fn))

    def read_csv(self):
        path = self.fn
        print(str(path))
        raw = pd.read_csv(path, usecols=self.headers)
        raw.dropna()
        raw = pd.get_dummies(data=raw, columns=['a_team', 'h_team', 'league'])
        # print(raw)
        # raw = .drop(['a_team', 'h_team', 'league'], axis=1)
        self.df = raw.copy()

    def chunk_df(self):
        self.df.sort_values(by='game_id', axis=1, inplace=True)
        self.df.set_index(keys=['game_id'], drop=False, inplace=True)
        games = self.df['game_id'].unique.tolist()

        print(games)

    def fit_data(self):
        transformer = RobustScaler().fit(self.df)
        transformer.transform(self.df)

    def reset(self):
        self.index = 0

    def next(self):
        if self.index >= len(self.df) - 1:
            return None, True

        values = self.df.iloc[self.index, 0:]

        cat_vals = values[12:].to_numpy().nonzero()

        self.index += 1

        return values, False

    def shape(self):
        return self.df.shape

    def a_odds(self):
        return self.df.ix[self.index, 'a_odds_ml']

    def h_odds(self):
        return self.df.ix[self.index, 'h_odds_ml']


class SipEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, file_name):
        self.file_name = file_name
        self.num = 1
        self.money = 100000
        self.bound = 16
        self.eq_a = 0
        self.eq_h = 0
        self.states = []
        self.state = None
        self.states.append(self.file_name)

        self.observation_space = spaces.Box(low=--100000000, high=100000000, shape=(12, ))
        self.action_space = spaces.Discrete(3)

        if len(self.states) == 0:
            raise NameError('Invalid empty directory {}'.format(self.file_name))

    def step(self, action):
        assert self.action_space.contains(action)

        portfolio = self.money + (self.eq_a * self.state.a_odds())

        a_odds = self.state.a_odds()
        h_odds = self.state.h_odds()

        a_eq = self.eq_calc(a_odds)
        h_eq = self.eq_calc(h_odds)

        prev_portfolio = self.money + self.eq_a + self.eq_h

        if action == ACTION_BUY_A:
            if self.money >= 100 * self.num and a_odds != 0:
                self.money -= 100 * self.num
                self.eq_a += self.num * a_eq
            else:
                print('forced skip')
        if action == ACTION_BUY_H:
            if self.money >= 100 * self.num and h_odds != 0:
                self.money -= 100 * self.num
                self.eq_h += self.num * h_eq
            else:
                print('forced skip')

        state, done = self.state.next()
        print(act_name(action))
        new_price = a_odds - h_odds
        if not done:
            new_price = self.state.a_odds()

        new_equity_price = new_price * self.eq_a
        reward = (self.money + self.eq_a + self.eq_h) - prev_portfolio
        print('reward: ' + str(reward))
        return state, reward, done, None

    def eq_calc(self, odd):
        if odd == 0:
            return 0
        if odd >= 100:
            return odd/100.
        elif odd < 100:
            return abs(100/odd)

    def reset(self):
        self.state = SippyState(random.choice(self.states))

        self.money = 0
        self.eq_a = 0
        self.eq_h = 0

        state, done = self.state.next()
        return state

    def _render(self, mode='human', close=False):
        pass


def act_name(act):
    if act == 0:
        return 'SKIP'
    elif act == 1:
        return 'BUY AWAY'
    elif act == 2:
        return 'BUY HOME'
