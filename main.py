import pandas as pd
import numpy as np
import os
import sys
import time
import logging
from logging.handlers import RotatingFileHandler
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from model.GRU_model import train as gru_train, predict as gru_predict
from model.LSTM_model import train as lstm_train, predict as lstm_predict

class Config:

    etf = "QQQ"
    
    feature_columns = list(range(1,7))
    label_columns = [2,3]
    label_in_feature_index = (lambda x,y: [x.index(i) for i in y])(feature_columns, label_columns)

    predict_day = 1

    model_type = "gru"
    dateset_type = 1
    train_data_rate = 0.95

    do_train = True
    do_predict = True
    time_step = 20
    valid_data_rate = 0.15
    random_seed = 42
    shuffle_train_data = True

    input_size = len(feature_columns)
    output_size = len(label_columns)
    hidden_size = 128       
    layers = 2
    dropout_rate = 0.2
    batch_size = 64
    learning_rate = 0.001
    epoch = 100
    do_continue_train = False
    do_train_visualized = False
    patience = 5

    model_save_path = "./saved_model/"
    if model_type=="gru":
        model_name = "GRU_model.pth"
    elif model_type=="lstm":
        model_name = "LSTM_model.pth"
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)

    data_save_path = "./dataset/"
    if not os.path.exists(data_save_path):
        os.makedirs(data_save_path)

def main(config):

    np.random.seed(config.random_seed)

    if config.dateset_type == 1:
        from dataset_QQQ import Data
        dataset = Data(config)
    elif config.dateset_type == 2:
        from dataset_top10 import Data
        dataset = Data(config)

    if config.do_train:
        train_X, valid_X, train_Y, valid_Y = dataset.get_train_and_valid_data()
        if config.model_type == "gru":
            gru_train(config, [train_X, train_Y, valid_X, valid_Y])
        elif config.model_type == "lstm":
            lstm_train(config, [train_X, train_Y, valid_X, valid_Y])

    if config.do_predict:
        test_X = dataset.get_test_data()
        if config.model_type == "gru":
            pred_result = gru_predict(config, test_X)
        elif config.model_type == "lstm":
            pred_result = lstm_predict(config, test_X)
    
    label_data = dataset.data[dataset.train_num + dataset.start_num_in_test : ,
                                            config.label_in_feature_index]
    predict_data = pred_result * dataset.std[config.label_in_feature_index] + \
                   dataset.mean[config.label_in_feature_index]

    label_name = [dataset.data_column_name[i] for i in config.label_in_feature_index]
    label_column_num = len(config.label_columns)

    loss = np.mean((label_data[config.predict_day:] - predict_data[:-config.predict_day] ) ** 2, axis=0)
    loss_norm = loss/(dataset.std[config.label_in_feature_index] ** 2)
    print("The mean squared error of stock {} is ".format(label_name) + str(loss_norm))

    label_X = range(dataset.data_num - dataset.train_num - dataset.start_num_in_test)
    predict_X = [ x + config.predict_day for x in label_X]

    for i in range(label_column_num):
        print("The predicted stock {} for the next {} day(s) is: ".format(label_name[i], config.predict_day) + str(np.squeeze(predict_data[-config.predict_day:, i])))

if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dateset_type", default=1, type=int, help="use which dataset type")
    parser.add_argument("-p", "--predict_day", default=1, type=int, help="predict how many days")
    parser.add_argument("-t", "--do_train", default=True, type=bool, help="whether to train")
    parser.add_argument("-m", "--model_type", default="gru", type=str, help="train with which model")
    args = parser.parse_args()
    config = Config()
    for key in dir(args):
        if not key.startswith("_"):
            setattr(config, key, getattr(args, key))
    main(config)