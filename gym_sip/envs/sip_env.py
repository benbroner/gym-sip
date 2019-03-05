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
        return int(self.game.iloc[self.index, 9])

    def h_odds(self):
        return int(self.game.iloc[self.index, 10])

    def game_over():
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

        self.hedges = []


    def step(self, action):  # action given to us from test.py
        prev_state = cur_state
        cur_state, done = self.game.next()
        state = cur_state - prev_state

        if not done:
            reward = self.action(action)

        return cur_state, reward, done, None


    def next(self):
        self.new_game()
        cur_state, done = self.game.next()
        return cur_state

    def reset(self):
        self.money = AUM
        self.next()

    def new_game(self):
        self.last_bet = None  # once a game has ended, bets are cleared 
        game_id = random.choice(list(self.games.keys()))
        return SippyState(self.games[game_id])

    def _reward(self, action):
        if action == ACTION_SKIP:
            return 0
        if self.last_bet == None:
            self.new_hedge(action)
        else:
            self.end_hedge(action)

    def new_hedge(self, action):
        if action == ACTION_BUY_A:
            bet_amt = h.bet_amt(self.money)

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
    def __init__(self, amt, team, a, h):
        self.amt = amt
        self.team = team  # 0 for away, 1 for home
        self.a_odds = a
        self.h_odds = h

    def __repr__(self)
        # simple console log of a bet
        print(h.act_name(self.team))
        print('bet amt: ' + str(self.amt))
        print('a_odds: ' + str(self.a_odds) + ' | h_odds: ' + str(self.h_odds))


class Hedge:
    def __init__(self, bet, bet2):
        self.net = h._net(bet, bet2)

    def __repr__(self):
        self.bet._print()
        self.bet2._print()
        print('hedged profit: ' + str(self.net))

    # TODO 
    # add function that writes bets to CSV for later analysis


