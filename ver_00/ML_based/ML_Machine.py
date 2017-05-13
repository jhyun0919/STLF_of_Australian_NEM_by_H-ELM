# -*- coding: utf-8 -*-

from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

import math
import time
import numpy as np
import pandas as pd

file_ = '/Users/JH/Desktop/NTU/NTU_Research/data/NEM_Load_Forecasting_Database.xls'

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
    """
    conduct preprocessing last after normalization
    :param data_present:
    :param temperature_max:
    :param temperature_mean:
    :param denominator:
    :return:
    """
    data_present = list(data_present) \
                   + list(preprocessing_filter(np.array(data_present), temperature_max, denominator)) \
                   + list(preprocessing_filter(np.array(data_present), temperature_mean, denominator))

    return np.array(data_present)


def data_alloter(df):
    """
    set data into following container format:
        dataset
            -Raw
                -Train
                    -feature
                    -target
                -Test
                    -feature
                    -target
            -Preprocessed
                -Train
                    -feature
                    -target
                -Test
                    -feature
                    -target
    :param df: data frame parsed from excel data
    :return: data allocated into the data container
    """
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

    dataset.Raw.Train.feature, dataset.Raw.Test.feature = data_splitter(raw_feature)
    dataset.Raw.Train.target, dataset.Raw.Test.target = data_splitter(raw_target)

    dataset.PreProcessed.Train.feature, dataset.PreProcessed.Test.feature = data_splitter(preprocessed_feature)
    dataset.PreProcessed.Train.target, dataset.PreProcessed.Test.target = data_splitter(preprocessed_target)

    return dataset


def train_multioutput_regressor(feature, target):
    """

    :param feature:
    :param target:
    :return:
    """
    start_train_time = time.time()
    regressor = MultiOutputRegressor(
        GradientBoostingRegressor(learning_rate=0.05, max_depth=1, random_state=0, verbose=0, n_estimators=10)).fit(feature, target)
    end_train_time = time.time()

    print str(regressor)
    print

    print "TrainingTime = ",
    print (end_train_time - start_train_time)
    print

    print "TrainingAccuracy = ",
    print str(RMSE(regressor.predict(feature), target))
    print

    return regressor


def test_multioutput_regressor(regressor, feature, target):
    """

    :param regressor:
    :param feature:
    :param target:
    :return:
    """
    start_test_time = time.time()
    predict_result = regressor.predict(feature)
    end_test_time = time.time()

    print "TestingTime = ",
    print (end_test_time - start_test_time)
    print

    print "TestingAccuracy = ",
    print str(RMSE(predict_result, target))
    print


def RMSE(predict, target):
    """
    calculate RMSE and return
    :param predict: predicted result data
    :param target: target data
    :return: RMSE score
    """
    error_sum = 0
    for i in range(0, len(predict)):
        error = mean_squared_error(predict[i], target[i]) ** 0.5
        error_sum += error
    return error_sum / len(predict)


def ML_MultiOutputRegression(train_data, test_data):
    # train regressor
    regressor = train_multioutput_regressor(train_data.feature, train_data.target)

    # test regressor
    test_multioutput_regressor(regressor, test_data.feature, test_data.target)


if __name__ == '__main__':
    df = pd.read_excel(file_, sheetname=NSW)
    dataset = data_alloter(df)

    # Raw Data
    ML_MultiOutputRegression(dataset.Raw.Train, dataset.Raw.Test)

    print
    print

    # Preprocessed Data
    ML_MultiOutputRegression(dataset.PreProcessed.Train, dataset.PreProcessed.Test)
