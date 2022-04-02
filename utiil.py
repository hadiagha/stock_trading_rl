import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import itertools
import argparse
import re

import pickle

from sklearn.preprocessing import StandardScaler

import os

import numpy as np
import pandas as pd
import itertools
import yfinance as yf

#Get data from api
def get_data(prd):
    stock_list = ['AAPL','MSI','SBUX']
    stock_data = pd.DataFrame()
    for sl in stock_list:
        data = yf.Ticker(sl).history(period=prd)
        data['stock'] = sl
        data = data.reset_index()
        data = data.loc[:,['Date','Close','stock']]
        stock_data = pd.concat([data,stock_data])

    stock_data=stock_data.pivot(index= 'Date',columns='stock', values='Close')
    stock_data = stock_data.reset_index()

    stock_data = stock_data[['AAPL','MSI','SBUX']]
    stock_data = stock_data.reset_index(drop=True)

    stock_data.to_csv('stock_data.csv',index=False)



