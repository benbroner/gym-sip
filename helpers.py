# helper functions for Sip OpenAI Gym environment
import sys
import pandas as pd


def csv(fn):
    # takes in file name, returns pandas dataframe
    # fn is type string
    df = pd.read_csv(fn)
    df.dropna()
    return df.copy()

def dummies(df, cols):
    # take in df and convert to df w dummies of cols 
    # df is pandas df, cols is string list
    one_hot_df = pd.get_dummies(data=df, columns=cols)
    return one_hot_df

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
        return 'skip'
    else: 
        return 'action outside of defined actions'


def _net(bet, bet2):
    # given two Bets, calculates the profit of the hedge
    # input: 2 Bet classes, output float

    if team == 0:
        return bet.amt * h._eq(bet.a_odds) - bet2.amt * h._eq(bet2.h_odds)
    elif team == 1:
        return bet.amt * h._eq(bet.h_odds) - bet2.amt * h._eq(bet2.a_odds)