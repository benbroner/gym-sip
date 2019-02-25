import gym
import random
import pandas as pd
from gym import spaces
from sklearn.preprocessing import RobustScaler

ACTION_SKIP = 0
ACTION_BUY_A = 1
ACTION_BUY_H = 2


class SippyState:
    def __init__(self, game):
        self.game = game  # df for game
        self.id = self.game.iloc[0, 2]  # first row, second column
        self.index = 0
        print(self.game['real_win'].values[1])

        print("Imported data from {}".format(self.id))

    def fit_data(self):
        transformer = RobustScaler().fit(self.game)
        transformer.transform(self.game)

    def reset(self):
        self.index = 0

    def next(self):
        if self.index >= len(self.game) - 1:
            return None, True

        values = self.game.iloc[self.index, 0:]

        self.index += 1

        return values, False

    def shape(self):
        return self.game.shape

    def a_odds(self):
        return int(self.game.iloc[self.index, 9])

    def h_odds(self):
        return int(self.game.iloc[self.index, 10])

    def num_games(self):
        return len(self.game['game_id'].unique())


class SipEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, file_name):

        self.fn = 'data/' + file_name + '.csv'
        self.df = None

        self.headers = ['a_team', 'h_team', 'league', 'game_id',
                        'a_pts', 'h_pts', 'secs', 'status', 'a_win', 'h_win', 'last_mod_to_start',
                        'num_markets', 'a_odds_ml', 'h_odds_ml', 'a_hcap_tot', 'h_hcap_tot']
        self.dtypes = {
                    'a_team': 'category',
                    'h_team': 'category',
                    'league': 'category',
                    'game_id': 'Int64',
                    'a_pts': 'Int16',
                    'h_pts': 'Int16',
                    'secs': 'Int16',
                    'status': 'Int16',
                    'a_win': 'Int16',
                    'h_win': 'Int16',
                    'last_mod_to_start': 'Float64',
                    'num_markets': 'Int16',
                    'a_odds_ml': 'Int32',
                    'h_odds_ml': 'Int32',
                    'a_hcap_tot': 'Int32',
                    'h_hcap_tot': 'Int32'
        }

        self.states = {}
        self.ids = []

        self.read_csv()
        self.chunk_df()
        self.wins()

        self.max_bets = 1  # MAX NUM OF HEDGED BETS. TOTAL BET COUNT = 2N

        self.a_bet_count = 0
        self.h_bet_count = 0

        self.bet_amt = 100
        self.money = 0  # DOESN'T MATTER IF IT RUNS OUT OF MONEY AND MAX BETS IS HELD CONSTANT

        self.eq_a = 0
        self.eq_h = 0
        self.a_odds = 0
        self.h_odds = 0
        self.adj_a_odds = 0
        self.adj_h_odds = 0

        new_id = random.choice(self.ids)
        self.state = SippyState(self.states[new_id])

        self.observation_space = spaces.Box(low=--100000000., high=100000000., shape=(537, ))
        self.action_space = spaces.Discrete(3)

        if len(self.states) == 0:
            raise NameError('Invalid empty directory {}'.format(self.fn))

    def step(self, action):
        assert self.action_space.contains(action)

        prev_portfolio = self.money

        self.actions(action)
        state, done = self.state.next()

        if not done:
            self.get_odds()
        if done:
            self.money -= self.bet_amt

        reward = self.money - prev_portfolio

        return state, reward, done, None

    def read_csv(self):
        raw = pd.read_csv(self.fn, usecols=self.headers)
        raw = raw.dropna()
        raw = pd.get_dummies(data=raw, columns=['a_team', 'h_team', 'league'])
        raw['real_win'] = -1
        self.df = raw.copy()

    def chunk_df(self):
        self.ids = self.df['game_id'].unique().tolist()
        self.states = {key: val for key, val in self.df.groupby('game_id')}

    def wins(self):
        print(str(len(self.ids)))
        for game_key in self.ids:
            game = self.states[game_key]
            self.win_set(game, game_key)
            # try:
            #     print(self.states[game_key]['real_win'])
            # except KeyError:
            #     continue
        print(str(len(self.ids)))

    def win_set(self, game, key):
        away_win = 0
        home_win = 0
        a_win = game['a_win'].unique().tolist()
        h_win = game['h_win'].unique().tolist()

        for elt in a_win:
            if elt == 1:
                away_win = 1
        for elt in h_win:
            if elt == 1:
                home_win = 1

        if (away_win == 1 and home_win == 1) or (away_win == 0 and home_win == 0):
            del self.states[key]
            self.ids.remove(key)
        elif away_win == 1:
            self.states[key]['real_win'] = 0
        else:
            self.states[key]['real_win'] = 1  # a home team win is 1

    def get_odds(self):
        self.a_odds = self.state.a_odds()
        self.h_odds = self.state.h_odds()
        self.adj_a_odds = eq_calc(self.a_odds)
        self.adj_h_odds = eq_calc(self.h_odds)

    def actions(self, action):
        if action == ACTION_BUY_A:
            if self.a_odds != 0 or self.a_bet_count < self.max_bets:
                self.money -= self.bet_amt
                self.eq_a += self.adj_a_odds
                self.a_bet_count += 1
        if action == ACTION_BUY_H:
            if self.h_odds != 0 or self.h_bet_count < self.max_bets:
                self.money -= self.bet_amt
                self.eq_h += self.adj_h_odds
                self.h_bet_count += 1

    def next(self):
        new_id = random.choice(self.ids)
        self.state = SippyState(self.states[new_id])

        self.eq_a = 0
        self.eq_h = 0

        state, done = self.state.next()
        return state

    def reset(self):
        new_id = random.choice(self.ids)
        self.state = SippyState(self.states[new_id])
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


def eq_calc(odd):
    if odd == 0:  # may cause bugs
        return 0
    if odd >= 100:
        return odd/100.
    elif odd < 100:
        return abs(100/odd)



