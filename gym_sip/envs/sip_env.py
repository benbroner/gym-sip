import gym
import random
import helpers as h

from sklearn.preprocessing import RobustScaler

# design element added march 4
# all functions that are arithmetic, assume the underscore '_' prefix
# eg def _eq(odd):


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

        self.ids()  # check to see if the games were not chunked correctly
        self.id = self.ids[0]

        self.index = self.game_len - 1

        # since the file was written append, we are decrementing from end
        self.values = self.game.iloc[self.index]

        print("imported {}".format(self.id))

    def next(self):

    def reset(self):
        self.index = len(self.game - 1)

    def shape(self):
        return self.game.shape

    def a_odds(self):
        return int(self.game.iloc[self.index, 9])

    def h_odds(self):
        return int(self.game.iloc[self.index, 10])

    def ids(self):
        ids = self.game['game_id'].unique()
        if len(ids) > 1:
            raise Exception('there was an error, chunked game has more than one id, the ids are {}'.format(ids))


class SipEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, fn):
        self.df = h.csv(fn)

    def step(self, action):

    def actions(self, action):

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
        self.eq_h = self.h_bet_amt * h._eq(self.init_h_odds)
        self.eq_a = self.a_bet_amt * h._eq(self.init_a_odds)

    def new_pair(self):
        if self.tot_bets % 2 == 1:
            self.init_a_odds = self.a_odds
            self.init_h_odds = self.h_odds
            self.init_a_amt = self.a_bet_amt
            self.init_h_amt = self.h_bet_amt

    def get_odds(self):
        self.a_odds = self.state.a_odds()
        self.h_odds = self.state.h_odds()
        self.adj_a_odds = h._eq(self.a_odds)
        self.adj_h_odds = h._eq(self.h_odds)

    def print_step(self):
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
    def __init__(self):
        self.amt = 0
        self.team = 0  # 0 for away, 1 for home
        self.a_odds = 0
        self.h_odds = 0

    def _print(self)
        # simple console log of a bet
        print(h.act_name(self.team))
        print('bet amt: ' + str(self.amt))
        print('a_odds: ' + str(self.a_odds) + ' | h_odds: ' + str(self.h_odds))


class Hedge:
    def __init__(self, bet, bet2):
        self.net = self._net()
        # might want to at least check the

    def _net(self):
        if team == 0:
            self.amt * h._eq(odd) -

    def _print(self):
        self.bet._print()
        self.bet2._print()
        self.


def print_bet(bet):
