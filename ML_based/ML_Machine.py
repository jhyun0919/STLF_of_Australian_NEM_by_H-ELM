# -*- coding: utf-8 -*-

from sklearn.datasets import make_regression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import GradientBoostingRegressor

import math
import numpy as np
import pandas as pd

file = '/Users/JH/Desktop/NTU/NTU_Research/data/NEM_Load_Forecasting_Database.xls'

concatenate_number = 25

QLD = 'Actual_Data_QLD'
NSW = 'Actual_Data_NSW'
VIC = 'Actual_Data_VIC'
SA = 'Actual_Data_SA'
TAS = 'Actual_Data_TAS'


# Set Classes as Data Container

class Structure:
    def __init__(self):
        self._feature = []
        self._target = []

    @property
    def feature(self):
        return self._feature

    @property
    def target(self):
        return self._target

    @feature.setter
    def feature(self, value):
        self._feature = value

    @target.setter
    def target(self, value):
        self._target = value


class Data:
    def __init__(self):
        pass

    class Train(Structure):
        def __init__(self):
            pass

    class Test(Structure):
        def __init__(self):
            pass


class DataSet:
    def __init__(self):
        pass

    class Raw:
        def __init__(self):
            pass

        class Train(Structure):
            def __init__(self):
                pass

        class Test(Structure):
            def __init__(self):
                pass

    class PreProcessed:
        def __init__(self):
            pass

        class Train(Structure):
            def __init__(self):
                pass

        class Test(Structure):
            def __init__(self):
                pass


# Set Functions

def normalization(data):
    return (data - min(data)) / (max(data) - min(data))


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
    return np.array(data[:splitter]), np.array(data[splitter + 1:])


def preprocessing_filter(data, nominator, denominator):
    return normalization(data) ** (nominator / denominator)


def preprocessing(data_present, temperature_max, temperature_mean, denominator):
    data_present = list(data_present) \
                   + list(preprocessing_filter(np.array(data_present), temperature_max, denominator)) \
                   + list(preprocessing_filter(np.array(data_present), temperature_mean, denominator))

    return np.array(data_present)


def data_alloter(df):
    dataset = DataSet()
    denominator = df['Mean Tem.'].min()

    raw_feature = []
    raw_target = []
    preprocessed_feature = []
    preprocessed_target = []

    for row in range(0, len(df)):
        # if both MaxTemp and MeanTemp are not nan
        if not math.isnan(df['Max Tem.'][row]) and not math.isnan(df['Mean Tem.'][row]):
            if not math.isnan(df['Max Tem.'][row + 1]) and not math.isnan(df['Mean Tem.'][row + 1]):
                powerload_present = normalization(np.array(df.loc[row][5:53]))
                powerload_future = normalization(np.array(df.loc[row + 1][5:53]))

                raw_feature.append(np.array(list(powerload_present)
                                            + list([df['Max Tem.'][row + 1]])
                                            + list([df['Mean Tem.'][row + 1]])))
                raw_target.append(np.array(powerload_future))

                preprocessed_powerload_present = preprocessing(powerload_present,
                                                               df['Max Tem.'][row + 1],
                                                               df['Mean Tem.'][row + 1],
                                                               denominator)

                preprocessed_feature.append(preprocessed_powerload_present)
                preprocessed_target.append(np.array(powerload_future))

    dataset.Raw.Train.feature, \
    dataset.Raw.Test.feature = data_splitter(raw_feature)
    dataset.Raw.Train.target, \
    dataset.Raw.Test.target = data_splitter(raw_target)

    dataset.PreProcessed.Train.feature, \
    dataset.PreProcessed.Test.feature = data_splitter(preprocessed_feature)
    dataset.PreProcessed.Train.target, \
    dataset.PreProcessed.Test.target = data_splitter(preprocessed_target)

    return dataset

if __name__ == '__main__':
    df = pd.read_excel(file, sheetname=QLD)

    dataset = data_alloter(df)

    X = dataset.Raw.Train.feature
    Y = dataset.Raw.Train.target

    print X
    print type(X)
    print X.shape
    print

    print Y
    print type(Y)
    print Y.shape
    print

    forecast = MultiOutputRegressor(GradientBoostingRegressor(random_state=0)).fit(X, Y)
    print forecast.predict(X)




