# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from sklearn import preprocessing

load_file = '/Users/JH/Documents/GitHub/PowerForecast/ver_01/data/NSW/load/NSW_NSLP_VICAGL_2006_2015.csv'

temperature_max_file = '/Users/JH/Documents/GitHub/PowerForecast/ver_01/data/NSW/weather/NSW_temp_max_2006_2015.csv'
temperature_min_file = '/Users/JH/Documents/GitHub/PowerForecast/ver_01/data/NSW/weather/NSW_temp_min_2006_2015.csv'
rainfall_file = '/Users/JH/Documents/GitHub/PowerForecast/ver_01/data/NSW/weather/NSW_rainfall_2006_2015.csv'
solar_expose_file = '/Users/JH/Documents/GitHub/PowerForecast/ver_01/data/NSW/weather/NSW_solar_2006_2015.csv'


# Set Functions

class Preprocess:
    def __init__(self):
        pass

    @staticmethod
    def data_splitter(data, ratio=0.8):
        """
        split data vector into training data & testing data
        :param data:
            data vector
        :param ratio:
            training data ratio
        :return:
            training_data vector, testing_data vector
        """
        splitter = int(len(data) * ratio)
        return np.array(data[:splitter]), np.array(data[splitter + 1:])

    @staticmethod
    def interpolation(data):
        """
        interpolate missing point
        :param data: 
            data vector
        :return:
            data vector after interpolation
        """
        for idx in xrange(0, len(data)):
            if np.isnan(data[idx]):
                data[idx] = data[idx - 1]
        return data

    @staticmethod
    def normalization(data):
        """
        normalize in range [−1,1]
        :param data:
            data
        :return:
            normalized data between -1 and 1
        """
        return (2 * ((data - min(data)) / (max(data) - min(data)))) - 1

    @staticmethod
    def normalization_load_data(data):
        """
        normalization for load data
        :param data: 
            load data before normalization
        :return: 
            load data after normalization
        """
        new_data = list()
        for row in xrange(0, len(data)):
            new_data.append(Preprocess.normalization(data[row]))

        return np.array(new_data)

    @staticmethod
    def normalization_weather_data(data):
        """
        normalization for weather data
        :param data:
            weather data before normalization
        :return:
            weather data after normalization
        """
        return Preprocess.normalization(preprocessing.scale(Preprocess.interpolation(data)))

    @staticmethod
    def integrate_weather_data(temp_max, temp_min, rainfall, solar):
        """
        integrate weather data into one numpy array
        :param temp_max: 
            temp_max vector
        :param temp_min:
            temp_min vector
        :param rainfall: 
            rainfall vector
        :param solar: 
            solar vector
        :return: 
            numpy array about weather
            (3652, 4)
        """
        return np.transpose(np.vstack((np.vstack((np.vstack((temp_max, temp_min)), rainfall)), solar)))

    @staticmethod
    def load_weather2feature_label(load, weather, day_diff):
        """
        
        :param load: 
            load vector
            (3652, 48)
        :param weather:
            weather vector
            (3652, 4)
        :param day_diff: 
        :return: 
            feature vector & label vector
        """
        feature_load_vector = list()
        feature_weather_vector = list()
        label_load_vector = list()
        if len(load) == len(weather):
            for row in xrange(0, (len(load) - day_diff)):
                feature_load_vector.append(load[row])
                feature_weather_vector.append(weather[row + day_diff])
                label_load_vector.append(load[row + day_diff])
        else:
            print 'two input vectors are having different length'
        return np.array(feature_load_vector), np.array(feature_weather_vector), np.array(label_load_vector)

    @staticmethod
    def filter_convolution(vector_0, vector_1):
        """
        exaggerate data with convolution
        :param vector_0: 
        :param vector_1: 
        :return: 
        """
        return np.convolve(vector_0, vector_1)

    @staticmethod
    def filter_outer_product(vector_0, vector_1):
        """
        exaggerate data with outer product
        :param vector_0: 
        :param vector_1: 
        :return: 
        """
        return np.reshape(np.outer(vector_0, vector_1), (1, 192))


class ReadCSV:
    def __init__(self):
        pass

    @staticmethod
    def laod(csv_file=load_file):
        """
        read csv file about load data and return as a numpy array
        :param csv_file: 
        :return: 
            (3652, 48)
        """
        return (pd.read_csv(csv_file)).values[:, 4:-1]

    @staticmethod
    def temp_max(csv_file=temperature_max_file):
        """
        read csv file about max temperature data and return as a numpy array
        :param csv_file: 
        :return: 
            (3652,)
        """
        return (pd.read_csv(csv_file)).values[:, 5]

    @staticmethod
    def temp_min(csv_file=temperature_min_file):
        """
        read csv file about min temperature data and return as a numpy array
        :param csv_file: 
        :return: 
            (3652,)
        """
        return (pd.read_csv(csv_file)).values[:, 5]

    @staticmethod
    def rainfall(csv_file=rainfall_file):
        """
        read csv file about rainfall data and return as a numpy array
        :param csv_file: 
        :return: 
            (3652,)
        """
        return (pd.read_csv(csv_file)).values[:, 5]

    @staticmethod
    def solar(csv_file=solar_expose_file):
        """
        read csv file about solar expose data and return as a numpy array
        :param csv_file: 
        :return: 
            (3652,)
        """
        return (pd.read_csv(csv_file)).values[:, 5]

    @staticmethod
    def load_data():
        """
        build train and test vector of load data
        :return: 
        """
        return Preprocess.data_splitter(Preprocess.normalization_load_data(ReadCSV.laod()))

    @staticmethod
    def weather_data():
        """
        build train and test vector of weather data
        :return: 
        """
        return Preprocess.data_splitter((Preprocess.integrate_weather_data(
            Preprocess.normalization_weather_data(ReadCSV.temp_max()),
            Preprocess.normalization_weather_data(ReadCSV.temp_min()),
            Preprocess.normalization_weather_data(ReadCSV.rainfall()),
            Preprocess.normalization_weather_data(ReadCSV.solar()))))


class DataAllocate:
    def __init__(self):
        pass

    @staticmethod
    def type00():
        """
        
        :return: 
        """
        load_train, load_test = ReadCSV.load_data()
        weather_train, weather_test = ReadCSV.weather_data()

        train_f_load, train_f_weather, train_l_load = Preprocess.load_weather2feature_label(load_train,
                                                                                            weather_train,
                                                                                            day_diff=1)
        test_f_load, test_f_weather, test_l_load = Preprocess.load_weather2feature_label(load_test,
                                                                                         weather_test,
                                                                                         day_diff=1)

        SaveData.elm_based(DataAllocate.filter00(train_f_load, train_f_weather, train_l_load), data_type='train00')
        SaveData.elm_based(DataAllocate.filter00(test_f_load, test_f_weather, test_l_load), data_type='test00')
        # SaveData.bp_based(train_feature, train_label, data_type='train00')
        # SaveData.bp_based(test_feature, test_label, data_type='test00')

    @staticmethod
    def type01():
        """
        
        :return: 
        """
        load_train, load_test = ReadCSV.load_data()
        weather_train, weather_test = ReadCSV.weather_data()

        train_f_load, train_f_weather, train_l_load = Preprocess.load_weather2feature_label(load_train,
                                                                                            weather_train,
                                                                                            day_diff=7)
        test_f_load, test_f_weather, test_l_load = Preprocess.load_weather2feature_label(load_test,
                                                                                         weather_test,
                                                                                         day_diff=7)

        SaveData.elm_based(DataAllocate.filter01(train_f_load, train_f_weather, train_l_load), data_type='train01')
        SaveData.elm_based(DataAllocate.filter01(test_f_load, test_f_weather, test_l_load), data_type='test01')
        # SaveData.bp_based(train_feature, train_label, data_type='train01')
        # SaveData.bp_based(test_feature, test_label, data_type='test01')

    @staticmethod
    def type02():
        """
        
        :return: 
        """
        load_train, load_test = ReadCSV.load_data()
        weather_train, weather_test = ReadCSV.weather_data()

        train_f_load, train_f_weather, train_l_load = Preprocess.load_weather2feature_label(load_train,
                                                                                            weather_train,
                                                                                            day_diff=7)
        test_f_load, test_f_weather, test_l_load = Preprocess.load_weather2feature_label(load_test,
                                                                                         weather_test,
                                                                                         day_diff=7)
        train_f_load_, train_f_weather_, train_l_load_ = Preprocess.load_weather2feature_label(load_train,
                                                                                               weather_train,
                                                                                               day_diff=1)
        test_f_load_, test_f_weather_, test_l_load_ = Preprocess.load_weather2feature_label(load_test,
                                                                                            weather_test,
                                                                                            day_diff=1)

        if len(train_f_load) == len(train_f_load_[6:]):
            SaveData.elm_based(DataAllocate.filter02(train_f_load, train_f_weather, train_f_load_[6:], train_l_load),
                               data_type='train02')
            SaveData.elm_based(DataAllocate.filter02(test_f_load, test_f_weather, test_f_load_[6:], test_l_load),
                               data_type='test02')
            # SaveData.bp_based(train_feature, train_label, data_type='train02')
            # SaveData.bp_based(test_feature, test_label, data_type='test02')
        else:
            print 'data allocation error in type02'

    @staticmethod
    def type03():
        """
        
        :return: 
        """
        load_train, load_test = ReadCSV.load_data()
        weather_train, weather_test = ReadCSV.weather_data()

        train_f_load, train_f_weather, train_l_load = Preprocess.load_weather2feature_label(load_train,
                                                                                            weather_train,
                                                                                            day_diff=7)
        test_f_load, test_f_weather, test_l_load = Preprocess.load_weather2feature_label(load_test,
                                                                                         weather_test,
                                                                                         day_diff=7)
        train_f_load_, train_f_weather_, train_l_load_ = Preprocess.load_weather2feature_label(load_train,
                                                                                               weather_train,
                                                                                               day_diff=1)
        test_f_load_, test_f_weather_, test_l_load_ = Preprocess.load_weather2feature_label(load_test,
                                                                                            weather_test,
                                                                                            day_diff=1)

        if len(train_f_load) == len(train_f_load_[6:]):
            SaveData.elm_based(DataAllocate.filter03(train_f_load, train_f_weather, train_f_load_[6:], train_l_load),
                               data_type='train03')
            SaveData.elm_based(DataAllocate.filter03(test_f_load, test_f_weather, test_f_load_[6:], test_l_load),
                               data_type='test03')
            # SaveData.bp_based(train_feature, train_label, data_type='train03')
            # SaveData.bp_based(test_feature, test_label, data_type='test03')
        else:
            print 'data allocation error in type03'

    @staticmethod
    def type04():
        """
        
        :return: 
        """
        load_train, load_test = ReadCSV.load_data()
        weather_train, weather_test = ReadCSV.weather_data()

        train_f_load, train_f_weather, train_l_load = Preprocess.load_weather2feature_label(load_train,
                                                                                            weather_train,
                                                                                            day_diff=7)
        test_f_load, test_f_weather, test_l_load = Preprocess.load_weather2feature_label(load_test,
                                                                                         weather_test,
                                                                                         day_diff=7)
        train_f_load_, train_f_weather_, train_l_load_ = Preprocess.load_weather2feature_label(load_train,
                                                                                               weather_train,
                                                                                               day_diff=1)
        test_f_load_, test_f_weather_, test_l_load_ = Preprocess.load_weather2feature_label(load_test,
                                                                                            weather_test,
                                                                                            day_diff=1)

        if len(train_f_load) == len(train_f_load_[6:]):
            SaveData.elm_based(DataAllocate.filter04(train_f_load, train_f_weather, train_f_load_[6:], train_l_load),
                               data_type='train04')
            SaveData.elm_based(DataAllocate.filter04(test_f_load, test_f_weather, test_f_load_[6:], test_l_load),
                               data_type='test04')
            # SaveData.bp_based(train_feature, train_label, data_type='train04')
            # SaveData.bp_based(test_feature, test_label, data_type='test04')
        else:
            print 'data allocation error in type04'

    @staticmethod
    def filter00(feature_load, feature_weather, label_load):
        """

        :param feature_load: 
        :param feature_weather: 
        :param label_load: 
        :return: 
        """
        input_vector = list()
        if (len(feature_load) == len(feature_weather)) and (len(feature_load) == len(label_load)):
            for row in xrange(0, len(label_load)):
                input_vector.append(np.append(label_load[row], np.append(feature_load[row], feature_weather[row])))
        else:
            print 'error in filter00'

        return np.array(input_vector)

    @staticmethod
    def filter01(feature_load, feature_weather, label_load):
        """

        :param feature_load: 
        :param feature_weather: 
        :param label_load: 
        :return: 
        """
        input_vector = list()
        if (len(feature_load) == len(feature_weather)) and (len(feature_load) == len(label_load)):
            for row in xrange(0, len(label_load)):
                input_vector.append(np.append(label_load[row], np.append(feature_load[row], feature_weather[row])))
        else:
            print 'error in filter01'

        return np.array(input_vector)

    @staticmethod
    def filter02(feature_load, feature_weather, feature_load_, label_load):
        """

        :param feature_load: 
        :param feature_weather: 
        :param feature_load_: 
        :param label_load: 
        :return: 
        """
        input_vector = list()
        if (len(feature_load) == len(feature_load_)) and (len(feature_load) == len(feature_weather)) and (
                    len(feature_load) == len(label_load)):
            for row in xrange(0, len(label_load)):
                input_vector.append(np.append(label_load[row],
                                              np.append(feature_load[row],
                                                        np.append(feature_load[row],
                                                                  feature_weather[row]))))
        else:
            print 'error in SaveData filter02'

        return np.array(input_vector)

    @staticmethod
    def filter03(feature_load, feature_weather, feature_load_, label_load):
        input_vector = list()
        if (len(feature_load) == len(feature_load_)) and (len(feature_load) == len(feature_weather)) and (
                    len(feature_load) == len(label_load)):
            for row in xrange(0, len(label_load)):
                input_vector.append(np.append(label_load[row], np.append(
                    Preprocess.filter_convolution(feature_load[row], feature_weather[row]),
                    Preprocess.filter_convolution(feature_load_[row], feature_weather[row]))))
        else:
            print 'error in SaveData filter03'

        return np.array(input_vector)

    @staticmethod
    def filter04(feature_load, feature_weather, feature_load_, label_load):
        input_vector = list()
        if (len(feature_load) == len(feature_load_)) and (len(feature_load) == len(feature_weather)) and (
                    len(feature_load) == len(label_load)):
            for row in xrange(0, len(label_load)):
                input_vector.append(np.append(label_load[row], np.append(
                    Preprocess.filter_outer_product(feature_load[row], feature_weather[row]),
                    Preprocess.filter_outer_product(feature_load_[row], feature_weather[row]))))
        else:
            print 'error in SaveData filter03'

        return np.array(input_vector)


class SaveData:
    def __init__(self):
        pass

    @staticmethod
    def elm_based(data_vector, data_type):
        with open('elm_' + data_type + '_data.csv', 'w') as csv_file:
            np.savetxt(csv_file, np.array(data_vector), delimiter=",")

    @staticmethod
    def bp_based():
        pass


if __name__ == '__main__':
    DataAllocate.type00()
    DataAllocate.type01()
    DataAllocate.type02()
    DataAllocate.type03()
    DataAllocate.type04()
