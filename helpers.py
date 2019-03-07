# helper functions for Sip OpenAI Gym environment
import sys
import datetime
import pandas as pd

def get_games(fn):
    # takes in fn and returns python dict of pd dfs 
    raw = csv(fn)
    df = one_hots(raw, ['sport', 'league', 'a_team', 'h_team'])
    df = dates(df)
    games = chunk(df, 'game_id')
    return games


def csv(fn):
    # takes in file name, returns pandas dataframe
    # fn is type string
    df = pd.read_csv(fn)
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
    df = df.drop(['lms_date', 'lms_time'], axis=1)
    return df


def chunk(df, col):
    # returns a python dict of pandas dfs, splitting the df arg by unique col value
    # df type pd df, col type string
    games = {key: val for key, val in df.groupby(col)}
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


def _act(act):
    # simple function to easily change the action number into a string
    # returns string
    if act == 0:
        return 'BOUGHT AWAY'
    elif act == 1:
        return 'BOUGHT HOME'
    elif act == 2:
        return 'SKIP'
    else: 
        return 'action outside of defined actions'


def _net(hedge):
    # given a bet pair (bet + hedge)
    # input: Hedge class, output float
    # env.is_valid() should have already caught zero odds lines
    # a full hedge equates the profit, so
    # bet.amt * _eq(bet.a) should be equal to bet2.amt * _eq(bet2.h)
    bet = hedge.bet
    bet2 = hedge.bet
    bet_sum = bet.amt + bet2.amt
    if bet.team == 0:
        return bet.amt * _eq(bet.a_odds) - bet_sum
    else:
        return bet.amt * _eq(bet.h_odds) - bet_sum


def _bet_amt(money):
    return 0.05 * money + 100


def _hedge_amt(bet, cur_odds):
    # takes in Bet 1 and calculates the 
    if bet.team == 0:
        return (bet.amt * _eq(bet.a_odds)) / _eq(cur_odds[1])
    else:
        return (bet.amt * _eq(bet.h_odds)) / _eq(cur_odds[0])