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


headers = [# 'a_team', 'h_team', 'sport', 'league', 
                'game_id', 'cur_time',
                'a_pts', 'h_pts', 'secs', 'status', 'a_win', 'h_win', 'last_mod_to_start',
                'num_markets', 'a_odds_ml', 'h_odds_ml', 'a_hcap_tot', 'h_hcap_tot']


def get_games(fn='data/nba2.csv'):
    # takes in fn and returns python dict of pd dfs 
    raw = csv(fn)
    print(raw)
    # df = one_hots(raw, ['sport', 'league', 'a_team', 'h_team'])
    # df = dates(raw)
    # df = df.drop(['lms_date', 'lms_time'], axis=1)  # remove if dates() is called
    # df = df.astype(np.float32)
    games = chunk(raw, 'game_id')
    return games

def remove_missed_wins(games):
    # takes in a dictionary of games 
    for g_id in list(games.keys()):
        if len(games[g_id]['a_win'].unique()) + len(games[g_id]['h_win'].unique()) != 3:
            del games[g_id]
    return games

def get_df(fn='/data/nba2.csv'):
    raw = csv(fn)
    df = one_hots(raw, ['sport', 'league', 'a_team', 'h_team'])
    df = dates(df)
    df = df.astype(np.float32)
    return df


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

def csv(fn):
    # takes in file name, returns pandas dataframe
    # fn is type string
    df = pd.read_csv(fn, usecols=headers)
    df.dropna()
    return df.copy()


def one_hots(df, cols):
    # take in df and convert to df w dummies of cols 
    # df is pandas df, cols is string list
    one_hot_df = pd.get_dummies(data=df, columns=cols)
    return one_hot_df


def dates(df):
    # convert ['lms_date', 'lms_time'] into datetimes
    df['datetime'] = df['lms_date'] + ' ' + df['lms_time']
    df['datetime'] = pd.to_datetime(df['datetime'], infer_datetime_format=True, errors='coerce')
    df['datetime'] = pd.to_numeric(df['datetime'])
    df = df.drop(['lms_date', 'lms_time'], axis=1)
    # df = df.drop(df['datetime'], axis=1)
    return df


def chunk(df, col):
    # returns a python dict of pandas dfs, splitting the df arg by unique col value
    # df type pd df, col type string
    games = {key: val for key, val in df.groupby(col)}
    print(len(games))
    games = remove_missed_wins(games)
    print(len(games))
    return games


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
        return 'HEDGED HOME'
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


def awaysale_price(bet, cur_odds):
    # takes in Bet 1 and calculates the 
        return  awaygraph_hedge_amt(cur_odds) * _eq(cur_odds[0]) - 100


def homesale_price(bet, cur_odds):
    if cur_odds[0] != None and cur_odds[1] != None:
        return homegraph_hedge_amt(cur_odds) * _eq(cur_odds[1]) - 100      
    else:
        return 0


def awaygraph_hedge_amt(cur_odds):
    return (100 * (abs(_eq(cur_odds[0])) + 1)) / (abs(_eq(cur_odds[1])) + 1)


def homegraph_hedge_amt(cur_odds): 
    return (100 * (abs(_eq(cur_odds[1])) + 1)) / (abs(_eq(cur_odds[0])) + 1)


def points_sum(a_points, h_points):
    # Simple sum at indices 0
    return int(a_points[0]) + int(h_points[0])
