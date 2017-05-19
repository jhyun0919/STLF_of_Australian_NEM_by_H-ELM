# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from sklearn import preprocessing

load_file = '/Users/JH/Documents/GitHub/PowerForecast/ver_01/data/NSW/load/NSW_NSLP_VICAGL_2006_2015.csv'

temperature_max_file = '/Users/JH/Documents/GitHub/PowerForecast/ver_01/data/NSW/weather/NSW_temp_max_2006_2015.csv'
temperature_min_file = '/Users/JH/Documents/GitHub/PowerForecast/ver_01/data/NSW/weather/NSW_temp_min_2006_2015.csv'
rainfall_file = '/Users/JH/Documents/GitHub/PowerForecast/ver_01/data/NSW/weather/NSW_rainfall_2006_2015.csv'
solar_expose_file = '/Users/JH/Documents/GitHub/PowerForecast/ver_01/data/NSW/weather/NSW_solar_2006_2015.csv'

CONV = 613
OUTER = 919


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
    def filter_convolution(vector_0, vector_1):
        """
        exaggerate data with convolution
        :param vector_0: 
        :param vector_1: 
        :return: 
        """
        # print vector_0.shape
        # print vector_1.shape
        return np.convolve(vector_0, vector_1)

    @staticmethod
    def filter_outer_product(vector_0, vector_1):
        """
        exaggerate data with outer product
        :param vector_0: 
        :param vector_1: 
        :return: 
        """
        return np.ravel(np.outer(vector_0, vector_1))


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
    def type_d(day_series_len):
        train_load, test_load = ReadCSV.load_data()
        train_weather, test_weather = ReadCSV.weather_data()

        combine_type = 'd' + str(day_series_len)
        SaveData.elm_based(DataAllocate.combine_d(train_load, train_weather, day_series_len), 'train', combine_type)
        SaveData.elm_based(DataAllocate.combine_d(test_load, test_weather, day_series_len), 'test', combine_type)

    @staticmethod
    def combine_d(load, weather, day_series_len):
        input_vector = list()
        if len(load) == len(weather):
            for row in xrange(0, len(load) - day_series_len + 1):
                if day_series_len == 1:
                    input_vector.append(np.append(load[row], weather[row]))
                elif day_series_len == 2:
                    input_vector.append(np.append(np.append(load[row], weather[row]), weather[row + 1]))
                elif day_series_len == 3:
                    input_vector.append(
                        np.append(np.append(np.append(load[row], weather[row]), weather[row + 1]), weather[row + 2]))
        else:
            print 'error in combine_d'
        return np.array(input_vector)

    @staticmethod
    def type_w(week_series_len):
        train_load, test_load = ReadCSV.load_data()
        train_weather, test_weather = ReadCSV.weather_data()

        combine_type = 'w' + str(week_series_len)
        SaveData.elm_based(DataAllocate.combine_w(train_load, train_weather, week_series_len), 'train', combine_type)
        SaveData.elm_based(DataAllocate.combine_w(test_load, test_weather, week_series_len), 'test', combine_type)

    @staticmethod
    def combine_w(load, weather, week_series_len):
        input_vector = list()
        day_series_len = week_series_len * 7
        if len(load) == len(weather):
            for row in xrange(0, len(load) - day_series_len + 1):
                if week_series_len == 1:
                    input_vector.append(np.append(load[row], weather[row]))
                elif week_series_len == 2:
                    input_vector.append(np.append(np.append(load[row], weather[row]), weather[row + 7]))
                elif week_series_len == 3:
                    input_vector.append(
                        np.append(np.append(np.append(load[row], weather[row]), weather[row + 7]), weather[row + 14]))
        else:
            print 'error in combine_w'
        return np.array(input_vector)

    @staticmethod
    def type_dw(day_series_len, week_series_len):
        train_load, test_load = ReadCSV.load_data()
        train_weather, test_weather = ReadCSV.weather_data()

        combine_type = 'd' + str(day_series_len) + 'w' + str(week_series_len)
        SaveData.elm_based(DataAllocate.combine_dw(train_load, train_weather, day_series_len, week_series_len), 'train',
                           combine_type)
        SaveData.elm_based(DataAllocate.combine_dw(test_load, test_weather, day_series_len, week_series_len), 'test',
                           combine_type)

    @staticmethod
    def combine_dw(load, weather, day_series_len, week_series_len):
        input_vector = list()
        if len(load) == len(weather):
            for row in xrange(0, len(load) - week_series_len * 7 + 1):
                day_vector = list()
                week_vector = list()

                if week_series_len == 1:
                    week_vector.append(weather[row])
                elif week_series_len == 2:
                    week_vector.append(np.append(weather[row], weather[row + 7]))
                elif week_series_len == 3:
                    week_vector.append(
                        np.append(np.append(weather[row], weather[row + 7]), weather[row + 14]))

                if day_series_len == 1:
                    day_vector.append(weather[row])
                elif day_series_len == 2:
                    day_vector.append(np.append(weather[row], weather[row + 1]))
                elif day_series_len == 3:
                    day_vector.append(
                        np.append(np.append(weather[row], weather[row + 1]), weather[row + 2]))

                input_vector.append(np.append(load[row], np.append(np.array(day_vector), np.array(week_vector))))

        else:
            print 'error in combine_dw'
        return np.array(input_vector)

    @staticmethod
    def type_dw_exag(day_series_len, week_series_len, filter_type):
        train_load, test_load = ReadCSV.load_data()
        train_weather, test_weather = ReadCSV.weather_data()

        combine_type = 'd' + str(day_series_len) + 'w' + str(week_series_len)
        SaveData.elm_based(
            DataAllocate.combine_dw_exag(train_load, train_weather, day_series_len, week_series_len, filter_type),
            'train',
            combine_type,
            filter_type)
        SaveData.elm_based(
            DataAllocate.combine_dw_exag(test_load, test_weather, day_series_len, week_series_len, filter_type),
            'test',
            combine_type,
            filter_type)

    @staticmethod
    def combine_dw_exag(load, weather, day_series_len, week_series_len, filter_type):
        input_vector = list()
        if len(load) == len(weather):
            for row in xrange(0, len(load) - week_series_len * 7 + 1):
                day_vector = list()
                week_vector = list()

                if week_series_len == 1:
                    week_vector.append(weather[row])
                elif week_series_len == 2:
                    week_vector.append(np.append(weather[row], weather[row + 7]))
                elif week_series_len == 3:
                    week_vector.append(
                        np.append(np.append(weather[row], weather[row + 7]), weather[row + 14]))

                if day_series_len == 1:
                    day_vector.append(weather[row])
                elif day_series_len == 2:
                    day_vector.append(np.append(weather[row], weather[row + 1]))
                elif day_series_len == 3:
                    day_vector.append(
                        np.append(np.append(weather[row], weather[row + 1]), weather[row + 2]))

                if filter_type == CONV:
                    input_vector.append(np.append(load[row], Preprocess.filter_convolution(day_vector, week_vector)))
                elif filter_type == OUTER:
                    input_vector.append(np.append(load[row], Preprocess.filter_outer_product(np.array(day_vector),
                                                                                             np.array(week_vector))))
        else:
            print 'error in combine_dw_exag'
        return np.array(input_vector)


class SaveData:
    def __init__(self):
        pass

    @staticmethod
    def elm_based(data_vector, data_type, combine_type, filter_type=None):
        if filter_type is None:
            with open('elm_' + data_type + '_' + combine_type + '_data.csv', 'w') as csv_file:
                np.savetxt(csv_file, np.array(data_vector), delimiter=",")
        elif filter_type == CONV:
            with open('elm_' + data_type + '_' + combine_type + 'conv' + '_data.csv', 'w') as csv_file:
                np.savetxt(csv_file, np.array(data_vector), delimiter=",")
        elif filter_type == OUTER:
            with open('elm_' + data_type + '_' + combine_type + 'outer' + '_data.csv', 'w') as csv_file:
                np.savetxt(csv_file, np.array(data_vector), delimiter=",")


if __name__ == '__main__':
    DataAllocate.type_dw_exag(1, 1, CONV)
    DataAllocate.type_dw_exag(2, 2, CONV)
    DataAllocate.type_dw_exag(3, 3, CONV)
    # DataAllocate.type_dw_exag(1, 1, OUTER)
    # DataAllocate.type_dw_exag(2, 2, OUTER)
    # DataAllocate.type_dw_exag(3, 3, OUTER)
