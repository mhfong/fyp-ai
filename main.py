import numpy as np
import argparse

from model.GRU_model import train as gru_train, predict as gru_predict
from model.LSTM_model import train as lstm_train, predict as lstm_predict
from dataset_QQQ import Data as dataset_QQQ, Config as Config_QQQ
from dataset_top10 import Data as dataset_top10, Config as Config_top10

def main(config):

    if config.dataset_type == "qqq":
        dataset = dataset_QQQ(config)
    elif config.dataset_type == "top10":
        dataset = dataset_top10(config)

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
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset_type", default="qqq", type=str, help="use which dataset type")
    parser.add_argument("-p", "--predict_day", default=1, type=int, help="predict how many days")
    parser.add_argument("-t", "--do_train", action='store_false', help="whether to train")
    parser.add_argument("-m", "--model_type", default="lstm", type=str, help="train with which model")
    args = parser.parse_args()
    dataset_type = args.dataset_type
    if dataset_type=="qqq":
        config = Config_QQQ()
    elif dataset_type=="top10":
        config = Config_top10()
    config_dict = {}
    for key in dir(args):
        if not key.startswith("_"):
            setattr(config, key, getattr(args, key))
    for key in dir(config):
        if not key.startswith("_"):
            config_dict[key] = getattr(config, key)
    for key, value in config_dict.items():
        print(key, ' : ', value)

    main(config)
