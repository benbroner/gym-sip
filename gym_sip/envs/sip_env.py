import gym
import random
import helpers as h
import numpy as np

# Macros for actions
ACTION_BUY_A = 0
ACTION_BUY_H = 1
ACTION_SKIP = 2

# Starting bank
AUM = 10000


class SippyState:
    # SippyState is a Gym state
    # Using pd df with unique 'game_id'

    def __init__(self, game):
        self.game = game  # store in State for repeated access
        self.game_len = len(game)  # used often
        self.id = self.ids()[0]
        self.index = 0
        # self.game_a_odds = self.game['a_odds']
        # self.game_h_odds = self.game['h_odds']
        # since the file was written append, we are decrementing from end
        self.cur_state = self.game.iloc[self.index]

        print("imported {}".format(self.id))

    def next(self):
        if self.game_over():
            return None, True

        # print(self.game.iloc[self.index, 0:])
        self.cur_state = self.game.iloc[self.index, 0:]

        if self.cur_state is None:
            return None, True

        self.index += 1
        return self.cur_state, False

    def reset(self):
        self.index = 0
        # self.index = len(self.game - 1)

    def shape(self):
        return self.game.shape

    def a_odds(self):
        return int(self.game.iloc[self.index, 12])
        # return int(self.game_a_odds[self.index])

    def h_odds(self):
        return int(self.game.iloc[self.index, 13])
        # return int(self.game_h_odds[self.index])

    def game_over(self):
        return self.index > (self.game_len - 1)

    def ids(self):
        ids = self.game['game_id'].unique()
        if len(ids) > 1:
            # check to see if the games were not chunked correctly
            raise Exception('there was an error, chunked game has more than one id, the ids are {}'.format(ids))
        return ids


class SipEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, fn):
        self.games = h.get_games(fn)
        self.game = None
        self.game_id = 0
        self.new_game()
        self.money = AUM
        self.last_bet = None  # 
        self.cur_state = self.game.cur_state  # need to store
        self.action = None
        self.hedges = []
        self.game_hedges = 0
        self.follow_bets = 0
        self.odds = ()  # storing current odds as 2-tuple
        self.action_space = gym.spaces.Discrete(3)
        self.observation_space = gym.spaces.Box(low=-100000000., high=100000000., shape=(self.game.shape()[1],),
                                                dtype=np.float32)

    def step(self, action):  # action given to us from test.py
        self.action = action   
        reward = 0 
        prev_state = self.cur_state

        self.cur_state, done = self.game.next()  # goes to the next timestep in current game
        # if self.cur_state is None:
        #     return None, 0, True, self.odds
        if done is True:
            del self.games[self.game_id]

        if done is True and self.last_bet is not None:
            print('forgot to bet')
            self.last_bet.__repr__()
            print(self.money)

            if prev_state[6] == 1 and self.last_bet.team == 0:  # if away won and last bet team is away
                reward = (h._eq(self.last_bet.a_odds) *  self.last_bet.amt) * 0.75

            elif prev_state[7] == 1 and self.last_bet.team == 1:  # if home won and last bet team is home
                reward = (h._eq(self.last_bet.h_odds) *  self.last_bet.amt) * 0.75
            else:
                reward = -self.last_bet.amt
            self.money += reward
            print(self.money)
            self.last_bet.__repr__()
            return None, reward, True, self.odds
        elif done is True:
            return None, 0, True, self.odds

        self._odds()

        if self.is_valid():
            reward = self.act()

        print(self.money)
        return self.cur_state, reward, done, self.odds

    def next(self):
        self.new_game()
        self.cur_state, done = self.game.next()
        return self.cur_state, done

    def reset(self):
        self.money = AUM
        return self.next()

    def new_game(self):
        self.game_hedges = 0
        self.follow_bets = 0
        self.last_bet = None  # once a game has ended, bets are cleared
        self.game_id = random.choice(list(self.games.keys()))
        self.game = SippyState(self.games[self.game_id])
        if self.game is None:
            del self.games[self.game_id]
            print('deleted a game')
            self.new_game()

    def act(self):
        if self.action == ACTION_SKIP:
            # print('skipped')
            return 0  # if skip, reward = 0
        elif self.last_bet is None:  # if last bet != None, then this bet is a hedge
            self._bet()
            return 0
        #elif self.action != self.last_bet.team:
        elif self.last_bet.team == self.action:
            if self.follow_bets < 4:
                self.last_bet
                new_amt = self.last_bet.amt + 100   # 100 is arbitrary
                self.last_bet.__repr__()
                if self.action == 0:
                    x = (h._eq(self.last_bet.a_odds) + h._eq(self.odds[0]))/2
                    new_odd = h.eq_to_odd(x)
                    self.last_bet.amt = new_amt
                    self.last_bet.odd = new_odd

                else:
                    x = (h._eq(self.last_bet.h_odds) + h._eq(self.odds[1]))/2
                    new_odd = h.eq_to_odd(x)
                    self.last_bet.amt = new_amt
                    self.last_bet.odd = new_odd
                self.follow_bets += 1
                print("lets fucking hammer this shit boi  anand you a dawg e sucks at math low key") 
                print(self.follow_bets)
                self.last_bet.__repr__()

        else:
            net = self._hedge()
            self.money += net
            
            return net

    def _bet(self):
        # we don't update self.money because we don't want it to get a negative reward on _bet()
        amt = h.bet_amt(self.money)
        self.last_bet = Bet(amt, self.action, self.odds, self.cur_state)
        self.last_bet.__repr__()

    def is_valid(self):
        # is_valid does NOT check for strict profit on hedge
        # it only checks for zero odds and if game is over
        if self.odds == (0, 0) or self.odds == None:
            return False
        else:
            return True

    def _hedge(self):
        hedge_amt = h.hedge_amt(self.last_bet, self.odds)
        hedged_bet = Bet(hedge_amt, self.action, self.odds, self.cur_state)
        net = h.net(self.last_bet, hedged_bet)
        if net > 0:
            hedge = Hedge(self.last_bet, hedged_bet)
            self.hedges.append(hedge)
            self.last_bet = None
            self.game_hedges += 1
            return hedge.net
        else:
            return 0

    def _odds(self):
        self.odds = (self.cur_state[10], self.cur_state[11])

    def get_state(self):
        return self.cur_state

    def __repr__(self):
        print('index in game: ' + str(self.cur_state.index))
        # TODO fix team name access and print hedge count
        # print('a team: ' + str() +
        #       ' | h_team ' + str())

    def _render(self, mode='human', close=False):
        pass


class Bet:
    # class storing bet info, will be stored in pair (hedged-bet)
    # might want to add time into game so we can easily aggregate when it is betting in the game
    # possibly using line numbers where they update -(1/(x-5)). x=5 is end of game

    # maybe bets should be stored as a step (csv line) and the bet amt and index into game.
    def __init__(self, amt, action, odds, cur_state):
        self.amt = amt
        self.team = action  # 0 for away, 1 for home
        self.a_odds = odds[0]
        self.h_odds = odds[1]
        self.cur_state = cur_state
        # self.__repr__()

    def __repr__(self):
        # simple console log of a bet
        print(h.act(self.team))
        print('bet amt: ' + str(self.amt) + ' | team: ' + str(self.team))
        print('a_odds: ' + str(self.a_odds) + ' | h_odds: ' + str(self.h_odds))


class Hedge:
    def __init__(self, bet, bet2):
        # input args is two Bets
        self.net = h.net(bet, bet2)
        self.bet = bet
        self.bet2 = bet2
        self.__repr__()

    def __repr__(self):
        self.bet.__repr__()
        self.bet2.__repr__()
        print('hedged profit: ' + str(self.net))
        print('\n')

    # TODO 
    # add function that writes bets to CSV for later analysis
