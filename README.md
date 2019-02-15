# gym-sip
Adapted from https://github.com/bioothod/gym-stocks
Huge thanks to bioothod.

OpenAI gym to play with basketball odds.

1. Download stock data in comma-separated CSV format, following fields are required `'last_mod_lines', 'a_odds_ml', 'h_odds_ml' into `data/` directory within this git sources, each separate datafile should have `.csv` extension

2. Install gym-sip
```
pip install --user -e .
```
3. Run example
```
import gym
import gym_sip
env = gym.make('Sip-v0')
print env.reset()
```

4. Initial (reset) conditions
You have 1000000 units of money and zero equity. Opeartion comission is 0.1%, there is no inflation (will be added if needed), i.e. negative reward per HOLD action.

gym-stocks opens one random csv file from `stocks` directory at every `reset()` call and yields one line per step. No normaization is being performed.

Every buy/sell action uses **previous** close price, i.e. not the price returned by the `step()` call, comission is being applied per each of these actions. Only **1** stock is being processed in these steps, i.e. if you select BUY, and the price is 50, only one stock will be bought (you will pay *50\*(1+0.1/100)*) even if you have more money.

If you do not have enough money to perform BUY action or you do not have equity to perform SELL action, nothing happens.

Portfolio cost equals to the sum of the money and equity times the closing price.
Reward is a difference between portfolio cost at the current and previous steps, reward calculation is being performed after money and equity have been updated (with the appropriate comission).
