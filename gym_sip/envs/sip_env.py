import gym
import random
import helpers as h

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
        self.index = self.game_len - 1

        # since the file was written append, we are decrementing from end
        self.cur_state = self.game.iloc[self.index]

        print("imported {}".format(self.id))

    def next(self):
        if self.game_over():
             return None, True

        self.cur_state = self.game.iloc[self.index, 0:]

        self.index -= 1
        return self.cur_state, False

    def reset(self):
        self.index = len(self.game - 1)

    def shape(self):
        return self.game.shape

    def a_odds(self):
        return int(self.game.iloc[self.index, 12])

    def h_odds(self):
        return int(self.game.iloc[self.index, 13])

    def game_over(self):
        return self.index < 0

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
        self.game = self.new_game()
        self.money = AUM
        self.last_bet = None  # 
        self.cur_state = None  # need to store 
        self.action = None
        self.hedges = []
        self.odds = ()  # storing current odds as 2-tuple


    def step(self, action):  # action given to us from test.py
        self.action = action    

        prev_state = self.cur_state

        self.cur_state, done = self.game.next()  # goes to the next timestep in current game
        state = self.cur_state - prev_state

        self._odds()

        if self.is_valid(done):
            if self.last_bet is not None and done is True:  # unhedged bet, lose bet1 amt
                reward = -self.last_bet.amt
            else:
                reward = 0  # 0 reward for non-valid bet
            return state, reward, done, None  
        else:
            reward = self.act()

        return state, reward, done, None

    def next(self):
        self.game = self.new_game()
        self.cur_state, done = self.game.next()
        return self.cur_state

    def reset(self):
        self.money = AUM
        self.next()

    def new_game(self):
        self.last_bet = None  # once a game has ended, bets are cleared 
        game_id = random.choice(list(self.games.keys()))
        return SippyState(self.games[game_id])

    def act(self):
        if self.action == ACTION_SKIP:
            return 0  # if skip, reward = 0

        if self.last_bet is None:  # if last bet != None, then this bet is a hedge
            self._bet()
            return 0
        else:
            net = self._hedge()
            self.money += net
            return net


    def _bet(self):
        bet_amt = h._bet_amt(self.money)
        self.last_bet = Bet(bet_amt, self.action, self.odds)
        # we don't update self.money because we don't want it to get a negative reward on _bet()

    def is_valid(self, done):
        # is_valid does NOT check for strict profit on hedge
        # it only checks for zero odds and if game is over
        if self.odds == (0, 0):
            return False
        elif done == True:
            return False
        else:
            return True

    def _hedge(self):
        hedge_amt = h._hedge_amt(self.last_bet, self.odds)
        hedged_bet = Bet(hedge_amt, self.action, self.odds)
        hedge = Hedge(self.last_bet, hedged_bet)
        self.hedges.append(hedge)
        return hedge.net

    def _odds(self):
        self.odds = (self.cur_state[12], self.cur_state[13])

    def __repr__(self):
        print('index in game: ' + str(self.state.index))
        print('a teams: ' + str(self.teams[self.id]['a_team'].iloc[0]) +
              ' | h_team ' + str(self.teams[self.id]['h_team'].iloc[0]))

    def _render(self, mode='human', close=False):
        pass


class Bet:
    # class storing bet info, will be stored in pair (hedged-bet)
    # might want to add time into game so we can easily aggregate when it is betting in the game
    # possibly using line numbers where they update -(1/(x-5)). x=5 is end of game

    # maybe bets should be stored as a step (csv line) and the bet amt and index into game.
    def __init__(self, amt, action, odds):
        self.amt = amt
        self.team = action  # 0 for away, 1 for home
        self.a_odds = odds[0]
        self.h_odds = odds[1]

    def __repr__(self)
        # simple console log of a bet
        print(h.act_name(self.team))
        print('bet amt: ' + str(self.amt))
        print('a_odds: ' + str(self.a_odds) + ' | h_odds: ' + str(self.h_odds))


class Hedge:
    def __init__(self, bet, bet2):
        # input args is two Bets
        self.net = h._net(bet, bet2)
        self.bet = bet
        self.bet2 = bet2

    def __repr__(self):
        self.bet._print()
        self.bet2._print()
        print('hedged profit: ' + str(self.net))

    # TODO 
    # add function that writes bets to CSV for later analysis

