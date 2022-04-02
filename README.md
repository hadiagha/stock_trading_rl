# stock_trading_rl
In this project, by using linear model, I have build an RL model to do trading for three stock index names: 'AAPL','MSI','SBUX'

The source for this tutorial is: Udemy.Artificial.Intelligence.Reinforcement.Learning
I am going to extend this model to other RL models and add more options to trading mechanism.

Frist of all create a new environment for your project
for example you can use this command in your anaconda prompt:

conda create -n rl_env

then you should move to the new env with this command:
conda activate rl_env

then run these commands:
conda config  --env --add channels conda-forge
conda config --env --set channel_priority strict

then you should install required libraries:
matplotlib
datetime
itertools
pickle
sklearn
os
numpy
pandas
itertools
yfinance

After all, you should select newly built env in your project all the time.

