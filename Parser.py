# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import math
import cPickle as pickle

file = '/Users/JH/Desktop/NTU/NTU_Research/data/NEM_Load_Forecasting_Database.xls'

concatenate_number = 13

QLD = 'Actual_Data_QLD'
NSW = 'Actual_Data_NSW'
VIC = 'Actual_Data_VIC'
SA = 'Actual_Data_SA'
TAS = 'Actual_Data_TAS'


# Set a Data Class

class DataSet:
    def __init__(self, train, test):
        self.train = train
        self.test = test


class Temperature(DataSet):
    def __init__(self, train, test):
        super(self.__class__, self).__init__(train, test)


class PowerLoad(DataSet):
    def __init__(self, train, test):
        super(self.__class__, self).__init__(train, test)


def normalize(array):
    return (array - min(array)) / (max(array) - min(array))


def data_splitter(data, ratio=0.8):
    """
    split data into training data & testing data
    :param data:
    :param ratio: training data ratio
    :return: train_data, test_data
    """
    splitter = int(len(data) * ratio)
    return data[:splitter], data[splitter + 1:]


def extract_temperature(df):
    max_scanner = []
    mean_scanner = []
    collector = []
    for row in xrange(0, len(df)):
        if not math.isnan(df['Max Tem.'][row]) and not math.isnan(df['Mean Tem.'][row]):
            max_scanner.append(df['Max Tem.'][row])
            mean_scanner.append(df['Mean Tem.'][row])
        if len(max_scanner) is concatenate_number:
            collector.append(normalize(np.array(max_scanner + mean_scanner)))
            max_scanner.pop(0)
            mean_scanner.pop(0)

    return data_splitter(collector)


def extract_powerload(df):
    pass


if __name__ == '__main__':
    df = pd.read_excel(file, sheetname=QLD)

    temperature_training, temperature_testing = extract_temperature(df)
    temperature_map = Temperature(temperature_training, temperature_testing)
    powerload_vector = PowerLoad(extract_powerload(df))

    pickle.dump(temperature_map, open('temperature_map.p', 'wb'))
