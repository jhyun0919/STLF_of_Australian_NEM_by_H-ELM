# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import math
import cPickle as pickle

file = '/Users/JH/Desktop/NTU/NTU_Research/data/NEM_Load_Forecasting_Database.xls'

concatenate_number = 25

QLD = 'Actual_Data_QLD'
NSW = 'Actual_Data_NSW'
VIC = 'Actual_Data_VIC'
SA = 'Actual_Data_SA'
TAS = 'Actual_Data_TAS'


# Set Classes as Data Container

class Data:
    def __init__(self, train, test):
        self.train = train
        self.test = test


class DataSet(object):
    class Temperature(Data):
        def __init__(self, train, test):
            super(self.__class__, self).__init__(train, test)

    class PowerLoad(Data):
        def __init__(self, train, test):
            super(self.__class__, self).__init__(train, test)


# Set Functions

def normalize(array):
    return (array - min(array)) / (max(array) - min(array))


def data_splitter(data, ratio=0.8):
    """
    split data into training data & testing data
    :param data:

    :param ratio:
        training data ratio
    :return:
        train_data, test_data
    """
    splitter = int(len(data) * ratio)
    return data[:splitter], data[splitter + 1:]


def extract_feature(df, dataset):
    """
    assign data to designed data container
    :param df:
        pandas data-frame read from excel data format
    :param dataset:
        defined data container
    :return:
        N/A
    """
    temperature_max_scanner = []
    temperature_mean_scanner = []
    temperature_collector = []

    powerload_scanner = []
    powerload_collector = []

    for row in xrange(0, len(df)):
        if not math.isnan(df['Max Tem.'][row]) and not math.isnan(df['Mean Tem.'][row]):
            temperature_max_scanner.append(df['Max Tem.'][row])
            temperature_mean_scanner.append(df['Mean Tem.'][row])

        if len(temperature_max_scanner) is concatenate_number:
            temperature_collector.append(normalize(np.array(temperature_max_scanner + temperature_mean_scanner)))
            temperature_max_scanner.pop(0)
            temperature_mean_scanner.pop(0)

            for col in xrange(5, 53):
                powerload_scanner.append(df.loc[row][col])
            powerload_collector.append(normalize(np.array(powerload_scanner)))
            del (powerload_scanner[:])

    dataset.Temperature.train, dataset.Temperature.test = data_splitter(np.array(temperature_collector))
    dataset.PowerLoad.train, dataset.PowerLoad.test = data_splitter(np.array(powerload_collector))


if __name__ == '__main__':
    df = pd.read_excel(file, sheetname=QLD)

    dataset = DataSet()

    extract_feature(df, dataset)

    print dataset.Temperature.train[0]
    print dataset.Temperature.train[0].shape # (concatenate_number*2, )
    print type(dataset.Temperature.train[0]) # numpy.ndarray

    print dataset.Temperature.train.shape # (data-length, concatenate_number*2)
    print type(dataset.Temperature.train) # numpy.ndarray

    print dataset.PowerLoad.train[0]
    print dataset.PowerLoad.train[0].shape # (48, )
    print type(dataset.PowerLoad.train[0]) # numpy.ndarray

    print dataset.PowerLoad.train.shape # (data-length, 48)
    print type(dataset.PowerLoad.train) # numpy.ndarray

    pickle.dump(dataset, open('DataSet.p', 'wb'))
