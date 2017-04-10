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


# pickling

def save_object(obj, filename):
    with open(filename, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)


"""
def extract_feature(df, dataset):

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

"""
if __name__ == '__main__':
    df = pd.read_excel(file, sheetname=QLD)

    dataset = data_alloter(df)

    print "dataset.Raw.Train.feature"
    print dataset.Raw.Train.feature
    print type(dataset.Raw.Train.feature)
    print dataset.Raw.Train.feature.shape
    print

    print "dataset.Raw.Train.feature[0]"
    print dataset.Raw.Train.feature[0]
    print type(dataset.Raw.Train.feature[0])
    print dataset.Raw.Train.feature[0].shape
    print

    print "dataset.Raw.Train.target"
    print dataset.Raw.Train.target
    print type(dataset.Raw.Train.target)
    print dataset.Raw.Train.target.shape
    print

    print "dataset.Raw.Train.target[0]"
    print dataset.Raw.Train.target[0]
    print type(dataset.Raw.Train.target[0])
    print dataset.Raw.Train.target[0].shape
    print

    print "dataset.Raw.Test.feature"
    print dataset.Raw.Test.feature
    print type(dataset.Raw.Test.feature)
    print dataset.Raw.Test.feature.shape
    print

    print "dataset.Raw.Test.feature[0]"
    print dataset.Raw.Test.feature[0]
    print type(dataset.Raw.Test.feature[0])
    print dataset.Raw.Test.feature[0].shape
    print

    print "dataset.Raw.Test.target"
    print dataset.Raw.Test.target
    print type(dataset.Raw.Test.target)
    print dataset.Raw.Test.target.shape
    print

    print "dataset.Raw.Test.target[0]"
    print dataset.Raw.Test.target[0]
    print type(dataset.Raw.Test.target[0])
    print dataset.Raw.Test.target[0].shape
    print

    # sample usage
    save_object(dataset, 'dataset.pkl')
