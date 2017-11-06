# gym-stocks
OpenAI gym to play with stock market data

1. Download stock data in comma-separated CSV format, following fields are required `'Date', 'Time', 'Open', 'High', 'Low', 'Close', 'Volume'` into `stocks/` directory within this git sources, each separate equity datafile should have `.csv` extension
2. Install gym-stocks
```
pip install --user -e .
```
3. Run example
```
import gym
import gym_stocks
env = gym.make('Stocks-v0')
print env.reset()
```
