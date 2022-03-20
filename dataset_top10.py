import pandas as pd
import numpy as np
import yfinance as yf
import requests
import re
import os
import sys
import time
import logging
from logging.handlers import RotatingFileHandler
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

'''
This function return all tickers of an ETF's holdings
Add these 3 var in class Config to use:
headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:83.0) Gecko/20100101 Firefox/83.0"}
url = "https://www.zacks.com/funds/etf/{}/holding"
holdings = get_holdings_name(headers,etf,url)

def get_holdings_name(headers, etf, url):
    with requests.Session() as req:
        req.headers.update(headers)
        r = req.get(url.format(etf))
        print(f"Extracting: {r.url}")
        holdings = re.findall(r'etf\\\/(.*?)\\', r.text)
    return sorted(holdings)
'''

class Config:
    etf = "QQQ"
    tickers = ["AAPL", "MSFT", "AMZN", "TSLA", "GOOG", "FB", "GOOGL", "NVDA", "PYPL", "ADBE"]

    data_save_path = "./df/"
    if not os.path.exists(data_save_path):
        os.makedirs(data_save_path)

def normalization():
    return 0

def main(config):
    tickers = config.tickers
    tickers.append(config.etf)
    for ticker in tickers:
        df = yf.download(ticker)
        df.to_csv(config.data_save_path+ticker+".csv")
        print(ticker,": ",len(df))


if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser()
    # parser.add_argument("-t", "--do_train", default=False, type=bool, help="whether to train")
    # parser.add_argument("-p", "--do_predict", default=True, type=bool, help="whether to train")
    # parser.add_argument("-b", "--batch_size", default=64, type=int, help="batch size")
    # parser.add_argument("-e", "--epoch", default=20, type=int, help="epochs num")
    args = parser.parse_args()
    config = Config()
    main(config)