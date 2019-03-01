import gym
import random
import pandas as pd
from gym import spaces
from sklearn.preprocessing import RobustScaler

ACTION_SKIP = 0
ACTION_BUY_A = 1

AUM = 10000


class SippyState:
    def __init__(self, game):
        self.game = game  # df for game
        self.ids = self.game['game_id'].unique()
        if len(self.ids) > 1:
            raise Exception('there was an error, chunked game has more than one id, the ids are {}'.format(self.ids))
        self.id = self.ids[0]
        self.index = 0
        self.init_h_odds()
        self.first_h_odd = self.h_odds()

        print("Imported data from {}".format(self.id))

    def fit_data(self):
        transformer = RobustScaler().fit(self.game)
        transformer.transform(self.game)

    def init_h_odds(self):
        while self.h_odds() == 0:
            self.next()

    def reset(self):
        self.index = 0

    def next(self):
        if self.index >= len(self.game) - 1:
            return None, True
        values = self.game.iloc[self.index, 0:]
        self.index += 1
        self.no_odds()
        return values, False

    def no_odds(self):
        while self.a_odds() == 0 and self.h_odds() == 0:
            print('moneyline closed')
            self.index += 1

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
        self.df = read_csv(self.fn, self.headers)
        self.states = {}
        self.ids = []
        self.ids, self.states = chunk_df(self.df)
        self.max_bets = 1  # MAX NUM OF HEDGED BETS. TOTAL BET COUNT = 2N
        self.a_bet_count = 0
        self.a_bet_amt = 0
        self.base_bet = 100
        self.money = AUM  # DOESN'T MATTER IF IT RUNS OUT OF MONEY AND MAX BETS IS HELD CONSTANT
        self.init_a_odds = 0
        self.init_h_odds = 0
        self.h_bet_amt = (0.05 * self.money) + self.base_bet  # it bottoms out if just 0.05 * self.money
        self.adj_a_odds = 0
        self.adj_h_odds = 0
        self.eq_h = 0
        self.a_odds = 0
        self.h_odds = 0
        self.reward_sum = 0
        self.pot_a_eq = 0
        self.state = None
        self.observation_space = spaces.Box(low=--100000000., high=100000000., shape=(self.df.shape[1], ))
        self.action_space = spaces.Discrete(2)
        if len(self.states) == 0:
            raise NameError('Invalid empty directory {}'.format(self.fn))

    def step(self, action):
        assert self.action_space.contains(action)
        prev_portfolio = self.money
        state, done = self.state.next()

        if not done:
            self.update()

        if done == 1 and self.a_bet_count == 0:
            print('forgot to hedge')
            self.money -= self.h_bet_amt + self.a_bet_amt

        self.actions(action)
        reward = self.money - prev_portfolio
        return state, reward, done, None

    def actions(self, action):
        sum = self.init_h_odds + self.a_odds
        if action == ACTION_BUY_A:
            if self.a_odds != 0 and self.a_bet_count < self.max_bets and sum > 0:
                self.a_bet_amt = (self.eq_h + self.h_bet_amt) / (self.adj_a_odds + 1)
                self.pot_a_eq = self.a_bet_amt * self.adj_a_odds
                self.money += self.pot_a_eq - self.h_bet_amt
                self.a_bet_count += 1
                self.print_info()
        if action == ACTION_SKIP:
            print('s')

    def next(self):
        self.new_game()
        self.update()
        state, done = self.state.next()
        self.init_h_odds = self.state.first_h_odd
        return state

    def reset(self):
        self.money = AUM
        self.next()

    def new_game(self):
        new_id = random.choice(self.ids)
        self.state = SippyState(self.states[new_id])
        self.a_bet_count = 0

    def update(self):
        self.get_odds()
        self.h_bet_amt = (0.05 * self.money) + self.base_bet
        self.eq_h = self.h_bet_amt * eq_calc(self.init_h_odds)

    def get_odds(self):
        self.a_odds = self.state.a_odds()
        self.h_odds = self.state.h_odds()
        self.adj_a_odds = eq_calc(self.a_odds)
        self.adj_h_odds = eq_calc(self.h_odds)

    def print_info(self):
        print('a_bet_amt: ' + str(self.a_bet_amt) + ' | h_bet_amt: ' + str(self.h_bet_amt))
        print('init_a_odds: ' + str(self.init_a_odds) + ' | init_h_odds: ' + str(self.init_h_odds))
        # print('a_odds: ' + str(self.state.a_odds()) + ' | h_odds: ' + str(self.state.h_odds()))
        print('pot_eq_a: ' + str(self.pot_a_eq) + ' | eq_h: ' + str(self.eq_h))

    def _render(self, mode='human', close=False):
        pass


def act_name(act):
    if act == 0:
        return 'SKIP'
    elif act == 1:
        return 'BUY AWAY'


def eq_calc(odd):
    if odd == 0:  # may cause bugs
        return 0
    if odd >= 100:
        return odd/100.
    elif odd < 100:
        return abs(100/odd)


def chunk_df(df):
    ids = df['game_id'].unique().tolist()
    states = {key: val for key, val in df.groupby('game_id')}
    return ids, states


def read_csv(fn, headers):
    raw = pd.read_csv(fn, usecols=headers)
    raw = raw.dropna()
    raw = pd.get_dummies(data=raw, columns=['a_team', 'h_team', 'league'])
    return raw.copy()
