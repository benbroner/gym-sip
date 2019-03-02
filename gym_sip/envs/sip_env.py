import gym
import random
import pandas as pd
from gym import spaces
from sklearn.preprocessing import RobustScaler

ACTION_SKIP = 0
ACTION_BUY_A = 1
ACTION_BUY_H = 2
AUM = 10000


class SippyState:
    def __init__(self, game):

        self.game = game  # df for game
        self.game_len = len(game)
        print("game has: " + str(self.game_len) + " lines")

        self.ids = self.game['game_id'].unique()
        if len(self.ids) > 1:
            raise Exception('there was an error, chunked game has more than one id, the ids are {}'.format(self.ids))
        self.id = self.ids[0]

        self.index = self.game_len - 1
        self.first_h_odd = self.h_odds()
        self.first_a_odd = self.a_odds()
        self.init_pos_a = self.init_a_odds()
        self.init_pos_h = self.init_h_odds()

        print("Imported data from {}".format(self.id))

    def fit_data(self):
        transformer = RobustScaler().fit(self.game)
        transformer.transform(self.game)

    def init_h_odds(self):
        i = self.game_len - 1
        while int(self.game.iloc[i, 10]) == 0 and self.first_h_odd == 0:
            i -= 1
        self.first_h_odd = int(self.game.iloc[i, 10])
        return i

    def init_a_odds(self):
        i = self.game_len - 1
        while int(self.game.iloc[i, 9]) == 0 and self.first_a_odd == 0:
            i -= 1
        self.first_a_odd = int(self.game.iloc[i, 9])
        return i

    def reset(self):
        self.index = len(self.game - 1)

    def next(self):
        if self.index < 0:
            return None, True
        values = self.game.iloc[self.index, 0:]
        self.index -= 1
        return values, False

    def shape(self):
        return self.game.shape

    def a_odds(self):
        return int(self.game.iloc[self.index, 9])

    def h_odds(self):
        return int(self.game.iloc[self.index, 10])


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
        self.df, self.teams = read_csv(self.fn, self.headers)
        self.num_games = num_games(self.df)
        self.states = {}
        self.id = 0
        self.ids = []
        self.ids, self.states = chunk_df(self.df)
        self.max_bets = 5  # MAX NUM OF HEDGED BETS. TOTAL BET COUNT = 2N
        self.a_bet_count = 0
        self.h_bet_count = 0
        self.a_bet_amt = 0
        self.base_bet = 100
        self.money = AUM  # DOESN'T MATTER IF IT RUNS OUT OF MONEY AND MAX BETS IS HELD CONSTANT
        self.init_a_odds = 0
        self.init_h_odds = 0
        self.init_a_amt = 0
        self.init_h_amt = 0
        self.h_bet_amt = (0.05 * self.money) + self.base_bet  # it bottoms out if just 0.05 * self.money
        self.adj_a_odds = 0
        self.adj_h_odds = 0
        self.tot_bets = 0
        self.eq_h = 0
        self.eq_a = 0
        self.a_odds = 0
        self.h_odds = 0
        self.reward_sum = 0
        self.pot_a_eq = 0
        self.pot_h_eq = 0
        self.state = None
        self.last_bet = None
        self.observation_space = spaces.Box(low=-100000000., high=100000000., shape=(self.df.shape[1], ))
        self.action_space = spaces.Discrete(3)
        if len(self.states) == 0:
            raise NameError('Invalid empty directory {}'.format(self.fn))

    def step(self, action):

        assert self.action_space.contains(action)
        prev_portfolio = self.money
        state, done = self.state.next()

        if not done:
            self.update()

        if done == 1 and self.h_bet_count > self.a_bet_count:
            print('forgot to hedge')
            self.money -= self.init_h_amt + self.pot_a_eq
        if done == 1 and self.h_bet_count < self.a_bet_count:
            print('forgot to hedge')
            self.money -= self.init_a_amt + self.pot_h_eq

        self.actions(action)
        reward = self.money - prev_portfolio
        self.reward_sum += reward
        return state, reward, done, None

    def actions(self, action):
        sum_for_a = self.init_h_odds + self.a_odds
        sum_for_h = self.init_a_odds + self.h_odds  # cant be zero

        if action == ACTION_BUY_A:
            if self.a_odds != 0 and self.h_odds != 0 and self.a_bet_count < self.max_bets:
                if self.last_bet == ACTION_BUY_A:
                    print('cant repeat bets, must hedge')
                    return
                self.a_bet_amt = (self.eq_h + self.h_bet_amt) / (self.adj_a_odds + 1)
                self.pot_a_eq = self.a_bet_amt * self.adj_a_odds
                self.money += self.pot_a_eq
                self.a_bet_count += 1
                self.tot_bets += 1
                self.new_pair()
                self.print_step()
                self.print_bet(action)
                self.last_bet = action

        if action == ACTION_BUY_H:
            if self.a_odds != 0 and self.h_odds != 0 and self.h_bet_count < self.max_bets:
                if self.last_bet == ACTION_BUY_H:
                    print('cant repeat bets, must hedge')
                    return
                self.h_bet_amt = (self.eq_a + self.a_bet_amt) / (self.adj_h_odds + 1)
                self.pot_h_eq = self.h_bet_amt * self.adj_h_odds
                self.money += self.pot_h_eq
                self.h_bet_count += 1
                self.tot_bets += 1
                self.new_pair()
                self.print_step()
                self.print_bet(action)
                self.last_bet = action

        if action == ACTION_SKIP:
            print('s')

    def new_game(self):
        self.id = random.choice(self.ids)
        self.state = SippyState(self.states[self.id])
        self.a_bet_count = 0
        self.h_bet_count = 0

    def next(self):
        self.new_game()
        self.update()
        state, done = self.state.next()
        self.init_h_odds = 0
        self.init_a_odds = 0
        return state

    def reset(self):
        self.money = AUM
        self.next()

    def update(self):
        self.get_odds()
        self.h_bet_amt = (0.05 * self.money) + self.base_bet
        self.a_bet_amt = self.h_bet_amt
        self.eq_h = self.h_bet_amt * eq_calc(self.init_h_odds)
        self.eq_a = self.a_bet_amt * eq_calc(self.init_a_odds)

    def new_pair(self):
        if self.tot_bets % 2 == 1:
            self.init_a_odds = self.a_odds
            self.init_h_odds = self.h_odds
            self.init_a_amt = self.a_bet_amt
            self.init_h_amt = self.h_bet_amt
            print(str(self.init_h_odds))

    def get_odds(self):
        self.a_odds = self.state.a_odds()
        self.h_odds = self.state.h_odds()
        self.adj_a_odds = eq_calc(self.a_odds)
        self.adj_h_odds = eq_calc(self.h_odds)

    def print_bet(self, action):
        print(act_name(action))
        print('a_bet_amt: ' + str(self.a_bet_amt) + ' | h_bet_amt: ' + str(self.h_bet_amt))
        print('init_a_odds: ' + str(self.init_a_odds) + ' | init_h_odds: ' + str(self.init_h_odds))
        print('eq_a: ' + str(self.pot_a_eq) + ' | eq_h: ' + str(self.eq_h))
        print('a_bet_count: ' + str(self.a_bet_count) + ' | h_bet_count: ' + str(self.h_bet_count))
        print('a_odds: ' + str(self.state.a_odds()) + ' | h_odds: ' + str(self.state.h_odds()))

    def print_step(self):
        print('index in game: ' + str(self.state.index))
        print('a teams: ' + str(self.teams[self.id]['a_team'].iloc[0]) +
              ' | h_team ' + str(self.teams[self.id]['h_team'].iloc[0]))

    def _render(self, mode='human', close=False):
        pass


def act_name(act):
    if act == 0:
        return 'SKIP'
    elif act == 1:
        return 'BOUGHT AWAY'
    else:
        return 'BOUGHT HOME'


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


def team_dict(df):
    teams = df.iloc[:, 1:4]
    teams_dict = {key: val for key, val in teams.groupby('game_id')}
    # print(teams_dict['a_team'])
    return teams_dict


def read_csv(fn, headers):
    raw = pd.read_csv(fn, usecols=headers)
    raw = raw.dropna()
    teams = team_dict(raw)
    raw = pd.get_dummies(data=raw, columns=['a_team', 'h_team', 'league'])
    return raw.copy(), teams


def num_games(df):
    num = len(df['game_id'].unique())
    print(str(num))
    return num
