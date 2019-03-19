# helper functions for Sip OpenAI Gym environment
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader

class Df(Dataset):
    def __init__(self, np_df, unscaled):
        self.data_len = len(np_df)
        self.data = np_df
        self.unscaled_data = unscaled
        print(self.data_len)

    def __getitem__(self, index):
        # line = self.data.iloc[index]
        line = self.data[index]
        line_tensor = torch.tensor(line)
        unscaled_line = self.unscaled_data[index]
        unscaled_tensor = torch.tensor(unscaled_line)        
        return line_tensor, unscaled_tensor

    def __len__(self):
        return self.data_len


headers = ['a_team', 'h_team', 'sport', 'league', 
                'game_id', 'cur_time',
                'a_pts', 'h_pts', 'secs', 'status', 'a_win', 'h_win', 'last_mod_to_start', 'last_mod_lines'

                'num_markets', 'a_odds_ml', 'h_odds_ml', 'a_hcap_tot', 'h_hcap_tot', 'game_start_time']


def get_games(fn='data/nba2.csv'):
    # takes in fn and returns python dict of pd dfs 
    df = get_df(fn)
    games = chunk(df, 'game_id')
    games = remove_missed_wins(games)
    return games


def get_df(fn='/data/nba2.csv'):
    raw = csv(fn)
    raw = drop_null_times(raw)
    df = one_hots(raw, ['sport', 'league', 'a_team', 'h_team'])
    df = df.drop(['lms_date', 'lms_time'], axis=1)
    return df


def chunk(df, col):
    # returns a python dict of pandas dfs, splitting the df arg by unique col value
    # df type pd df, col type string
    games = {key: val for key, val in df.groupby(col)}
    return games


def csv(fn):
    # takes in file name string, returns pandas dataframe
    df = pd.read_csv(fn)
    return df.copy()


def one_hots(df, cols):
    # df is pandas df, cols is string list
    one_hot_df = pd.get_dummies(data=df, columns=cols, sparse=True)
    return one_hot_df


def remove_missed_wins(games):
    # takes in a dictionary of games 
    for g_id in list(games.keys()):
        if len(games[g_id]['a_win'].unique()) + len(games[g_id]['h_win'].unique()) != 3:
            del games[g_id]
    return games


def drop_null_times(df, columns=['lms_date', 'lms_time']):
    # given pandas df and list of strings for columns. convert '0' values to np.datetime64
    init_len = len(df)
    
    print('dropping null times from columns: {}'.format(columns))
    print('df init length: {}'.format(init_len))
    
    for col in columns:
        df[col] = df[col].replace('0', np.nan)
        # df[col] = pd.to_datetime(df[col])

    df = df.dropna()

    after_len = len(df)
    delta = init_len - after_len

    print('df after length: {}'.format(after_len))
    print('delta (lines removed): {}'.format(delta))
    return df


def apply_min_game_len(games, min_lines=500):
    # given dict of game dataframes and an integer > 0 for the minimum length of a game in csv lines 
    print('applying minimum game len of : {}'.format(min_lines))
    print('before apply: {}'.format(len(games)))
    for key, value in games.copy().items():
        game_len = len(value)
        if game_len < min_lines:
            print('deleted game_id: {}'.format(key))
            print('had length: {}'.format(game_len))
            del games[key]
    print('after apply: {}'.format(len(games)))
    return games


def df_info(df):
    # TODO
    # given pd df, return the general important info in console
    # num games, teams, etc 
    pass


def label_split(df, col):
    # give column to be predicted given all others in csv
    # df is pd, col is string
    Y = df[col]
    X = df
    return X, Y


def train_test(df, train_frac=0.6):
    # TODO
    pass


def teams_given_state(state):  
    # given np array, representing a state (csv_line). returns tuple of teams
    

    return state


def dates(df):
    # convert ['lms_date', 'lms_time'] into datetimes
    df['datetime'] = df['lms_date'] + ' ' + df['lms_time']
    df['datetime'] = pd.to_datetime(df['datetime'], infer_datetime_format=True, errors='coerce')
    df['datetime'] = pd.to_numeric(df['datetime'])
    df = df.drop(['lms_date', 'lms_time'], axis=1)
    # df = df.drop(df['datetime'], axis=1)
    return df


def _eq(odd):
    # to find the adjusted odds multiplier 
    # returns float
    if odd == 0:
        return 0
    if odd >= 100:
        return odd/100.
    elif odd < 100:
        return abs(100/odd)


def eq_to_odd(equity):
    if equity > 1:
        odd = 100 * equity
        return odd
    elif equity <= 1:
        odd = -100/equity
        return odd
    elif equity == 0:
        return 0


def act(a):
    # simple function to easily change the action number into a string
    # returns string
    if a == 0:
        return 'BOUGHT AWAY'
    elif a == 1:
        return 'BOUGHT HOME'
    elif a == 2:
        return 'SKIP'
    else: 
        return 'action outside of defined actions'


def net(bet, bet2):
    # given a bet pair (bet + hedge)
    # input: Hedge class, output float
    # env.is_valid() should have already caught zero odds lines
    # a full hedge equates the profit, so
    # bet.amt * _eq(bet.a) should be equal to bet2.amt * _eq(bet2.h)
    bet_sum = bet.amt + bet2.amt
    if bet.team == 0:
        return bet.amt * _eq(bet.a_odds) - bet2.amt
    else:
        return bet.amt * _eq(bet.h_odds) - bet2.amt


def bet_amt(money):
    # return 0.05 * money + 100  # 100 is arbitrary
    return 100


def hedge_amt(bet, cur_odds):
    # takes in Bet 1 and calculates the 
    if bet.team == 0:
        return (bet.amt * (_eq(bet.a_odds) + 1))/ (_eq(cur_odds[1]) + 1)
    else:
        return (bet.amt * (_eq(bet.h_odds) + 1)) / (_eq(cur_odds[0]) + 1)


def net_given_odds(bet, cur_odds):
    bet2_amt = hedge_amt(bet, cur_odds)
    bet_sum = bet.amt + bet2_amt
    if bet.team == 0:
        return bet.amt * _eq(bet.a_odds) - bet2_amt
    else:
        return bet.amt * _eq(bet.h_odds) - bet2_amt
