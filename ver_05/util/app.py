# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from sklearn import preprocessing
from datetime import date
from datetime import timedelta
import os.path

load_files_list = ['/Users/JH/Documents/GitHub/PowerForecast/ver_05/data/load/NSW_nslp2006.csv',
                   '/Users/JH/Documents/GitHub/PowerForecast/ver_05/data/load/NSW_nslp2007.csv',
                   '/Users/JH/Documents/GitHub/PowerForecast/ver_05/data/load/NSW_nslp2008.csv',
                   '/Users/JH/Documents/GitHub/PowerForecast/ver_05/data/load/NSW_nslp2009.csv',
                   '/Users/JH/Documents/GitHub/PowerForecast/ver_05/data/load/NSW_nslp2010.csv',
                   '/Users/JH/Documents/GitHub/PowerForecast/ver_05/data/load/NSW_nslp2011.csv',
                   '/Users/JH/Documents/GitHub/PowerForecast/ver_05/data/load/NSW_nslp2012.csv',
                   '/Users/JH/Documents/GitHub/PowerForecast/ver_05/data/load/NSW_nslp2013.csv',
                   '/Users/JH/Documents/GitHub/PowerForecast/ver_05/data/load/NSW_nslp2014.csv',
                   '/Users/JH/Documents/GitHub/PowerForecast/ver_05/data/load/NSW_nslp2015.csv']

load_areas = ['ACTEWAGL', 'CITIPOWER', 'COUNTRYENERGY', 'ENERGYAUST', 'INTEGRAL', 'POWERCOR', 'TXU', 'UMPLP', 'UNITED',
              'VICAGL']

temperature_max_file = '/Users/JH/Documents/GitHub/PowerForecast/ver_05/data/weather/NSW_temp_max_2006_2015.csv'
temperature_min_file = '/Users/JH/Documents/GitHub/PowerForecast/ver_05/data/weather/NSW_temp_min_2006_2015.csv'
rainfall_file = '/Users/JH/Documents/GitHub/PowerForecast/ver_05/data/weather/NSW_rainfall_2006_2015.csv'
solar_expose_file = '/Users/JH/Documents/GitHub/PowerForecast/ver_05/data/weather/NSW_solar_2006_2015.csv'

START_DATE = date(2006, 1, 1)

WEATHER_DAILY = 'wd'
LOAD_WEATHER_DAILY = 'lwd'
LOAD_WEATHER_WEEKLY = 'lww'
LOAD_WEATHER_DAILY_WEEKLY = 'lwdw'

SAVE_PATH = '/Users/JH/Documents/GitHub/PowerForecast/ver_05/data/input_vector'


# Set Functions

class Preprocess:
    def __init__(self):
        pass

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
    def normalization(data, a, b):
        """
        normalize in range [a,b]
        :param data: 
        :param a: 
        :param b: 
        :return: 
            normalized data
        """
        return ((b - a) * ((data - min(data)) / (max(data) - min(data)))) + a

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
            new_data.append(Preprocess.normalization(data[row], 0, 1))

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
        new_data = list()
        for col in xrange(0, len(data[0, :])):
            new_data.append(
                Preprocess.normalization(preprocessing.scale(Preprocess.interpolation(data[:, col])), 0.5, 1.5))
        new_data = np.array(new_data)
        return np.transpose(new_data)

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
        return np.array(data[:splitter]), np.array(data[splitter:])

    @staticmethod
    def data_concatenate(data_0, data_1):
        """
        concatenate two input data
        :param data_0: 
        :param data_1: 
        :return: 
            one conatenated data
        """
        return np.append(data_0, data_1)

    @staticmethod
    def power_function(load_vector, weather_vector):
        """
        load data powered by weather data
        :param load_vector: 
        :param weather_vector: 
        :return: 
        """
        input_vector = list()

        powered_temp_max = np.power(load_vector, weather_vector[0])
        powered_temp_min = np.power(load_vector, weather_vector[1])
        powered_temp_rain = np.power(load_vector, weather_vector[2])
        powered_temp_solar = np.power(load_vector, weather_vector[3])

        temp_row_vector = Preprocess.data_concatenate(load_vector, powered_temp_max)
        temp_row_vector = Preprocess.data_concatenate(temp_row_vector, powered_temp_min)
        temp_row_vector = Preprocess.data_concatenate(temp_row_vector, powered_temp_rain)
        temp_row_vector = Preprocess.data_concatenate(temp_row_vector, powered_temp_solar)

        input_vector.append(temp_row_vector)

        return np.array(input_vector)


class ReadCSV:
    def __init__(self):
        pass

    @staticmethod
    def load(csv_files=load_files_list):
        """
        read load csv data and return into a dictionary
        :param csv_files: 
        :return: 
            load dictionary
        """
        load_dict = dict()

        for file_num in xrange(0, len(csv_files)):
            df = pd.read_csv(csv_files[file_num])
            for area_num in xrange(0, len(load_areas)):
                temp_load = (df.loc[df['ProfileArea'] == load_areas[area_num]]).ix[:, 4:52]
                if file_num == 0:
                    load_dict[load_areas[area_num]] = np.array(temp_load)
                else:
                    load_dict[load_areas[area_num]] = np.vstack((load_dict[load_areas[area_num]], temp_load))

        return load_dict

    @staticmethod
    def weather():
        """
        build a numpy array of weather data
        :return: 
            weather numpy array
        """
        return Preprocess.integrate_weather_data(ReadCSV.temp_max(), ReadCSV.temp_min(), ReadCSV.rainfall(),
                                                 ReadCSV.solar())

    @staticmethod
    def temp_max(csv_file=temperature_max_file):
        """
        read csv file about max temperature data and return as a numpy array
        :param csv_file: 
        :return: 
            max temerature numpy array
        """
        return (pd.read_csv(csv_file)).values[:, 5]

    @staticmethod
    def temp_min(csv_file=temperature_min_file):
        """
        read csv file about min temperature data and return as a numpy array
        :param csv_file: 
        :return: 
            min temperature numpy array
        """
        return (pd.read_csv(csv_file)).values[:, 5]

    @staticmethod
    def rainfall(csv_file=rainfall_file):
        """
        read csv file about rainfall data and return as a numpy array
        :param csv_file: 
        :return: 
            rain fall numpy array
        """
        return (pd.read_csv(csv_file)).values[:, 5]

    @staticmethod
    def solar(csv_file=solar_expose_file):
        """
        read csv file about solar expose data and return as a numpy array
        :param csv_file: 
        :return: 
            solar expose numpy array
        """
        return (pd.read_csv(csv_file)).values[:, 5]


class Numpy2Pandas:
    def __init__(self):
        pass

    @staticmethod
    def set_date_list(start_date, date_list_length):
        """
        set a date list
        :param start_date: 
        :param date_list_length: 
        :return: 
            date list
        """
        date_list = list()
        present_date = start_date
        for idx in xrange(0, date_list_length):
            date_list.append(present_date)
            present_date = present_date + timedelta(days=1)
        return np.array(date_list)

    @staticmethod
    def add_date_index(data):
        """
        convert numpy array to pandas data frame & add date index to given numpy array
        :param data: 
        :return: 
            pandas data frame with date index
        """
        start_date = START_DATE
        if type(data) is dict:
            area_names = sorted(data.iterkeys())
            for idx in xrange(0, len(area_names)):
                date_list = Numpy2Pandas.set_date_list(start_date, len(data[area_names[idx]]))
                data[area_names[idx]] = data[area_names[idx]].set_index(date_list)
        else:
            date_list = Numpy2Pandas.set_date_list(start_date, len(data))
            data = data.set_index(date_list)

        return data

    @staticmethod
    def add_number_index(data):
        data.index = range(1, len(data) + 1)
        return data

    @staticmethod
    def acc_data2df(acc_data):
        df = pd.DataFrame(acc_data, columns=['RMSE'])
        return df

    @staticmethod
    def load_data2df(load_data):
        """
        convert load numpy array to pandas data frame & add date index to given load numpy array
        :param load_data: 
        :return: 
            pandas load data frame with date index
        """
        data_dictionary = dict()
        area_names = sorted(load_data.iterkeys())
        for area_num in xrange(0, len(area_names)):
            temp_dictionary = dict()
            for idx in xrange(0, 48):
                key_name = idx + 1
                temp_dictionary[key_name] = load_data[area_names[area_num]][:, idx]
            data_dictionary[area_names[area_num]] = pd.DataFrame(temp_dictionary)
        return Numpy2Pandas.add_date_index(data_dictionary)

    @staticmethod
    def weather_data2df(weather_data):
        """
        convert weather numpy array to pandas data frame & add date index to given weather numpy array
        :param weather_data: 
        :return: 
            pandas weather frame with date index
        """
        data_dictionary = dict()
        weather_key = ['Temperature_Max', 'Temperature_min', 'Rain_fall', 'Solar_exposure']
        for idx in xrange(0, 4):
            data_dictionary[weather_key[idx]] = weather_data[:, idx]
        data_dictionary = pd.DataFrame(data_dictionary)

        return Numpy2Pandas.add_date_index(data_dictionary)


class DataAllocate:
    def __init__(self):
        pass

    class STLF:
        def __init__(self):
            pass

        @staticmethod
        def allocator(load_vector, weather_vector, filter_type):
            """
            set training and testing input vectors with defined filters
            :param load_vector: 
            :param weather_vector: 
            :param filter_type: 
            :return:
                constructed training and testing input vector
            """
            train_load, test_load = Preprocess.data_splitter(load_vector)
            train_weather, test_weather = Preprocess.data_splitter(weather_vector)

            if filter_type is WEATHER_DAILY:
                train_input_vector = DataAllocate.Filter.filter_weather_daily(train_load, train_weather)
                test_input_vector = DataAllocate.Filter.filter_weather_daily(test_load, test_weather)
                return train_input_vector, test_input_vector
            elif filter_type is LOAD_WEATHER_DAILY:
                train_input_vector = DataAllocate.Filter.filter_load_weather_daily(train_load, train_weather)
                test_input_vector = DataAllocate.Filter.filter_load_weather_daily(test_load, test_weather)
                return train_input_vector, test_input_vector
            elif filter_type is LOAD_WEATHER_WEEKLY:
                train_input_vector = DataAllocate.Filter.filter_load_weather_weekly(train_load, train_weather)
                test_input_vector = DataAllocate.Filter.filter_load_weather_weekly(test_load, test_weather)
                return train_input_vector, test_input_vector
            elif filter_type is LOAD_WEATHER_DAILY_WEEKLY:
                train_input_vector = DataAllocate.Filter.filter_load_weather_daily_weekly(train_load, train_weather)
                test_input_vector = DataAllocate.Filter.filter_load_weather_daily_weekly(test_load, test_weather)
                return train_input_vector, test_input_vector

            else:
                print 'error during deciding the type of filter'

    class Filter:
        def __init__(self):
            pass

        @staticmethod
        def filter_weather_daily(load_vector, weather_vector):
            """
            set input label and feature vector as following
                label = load(k)
                feature = weather(k)
            :param load_vector: 
            :param weather_vector: 
            :return: 
                constructed input_vector
            """
            input_vector = list()
            for row in xrange(0, len(load_vector)):
                feature = load_vector[row]
                label = weather_vector[row]

                input_vector.append(Preprocess.data_concatenate(feature, label))
            return np.array(input_vector)

        @staticmethod
        def filter_load_weather_daily(load_vector, weather_vector):
            """
            set input label and feature vector as following
                label = load(k)
                feature = load(k-1), weather(k)
            :param load_vector: 
            :param weather_vector: 
            :return: 
                constructed input_vector
            """
            input_vector = list()
            for row in xrange(1, len(load_vector)):
                load_k = load_vector[row]
                load_k_day_ago = load_vector[row - 1]
                weather_k = weather_vector[row]

                feature = load_k
                label = Preprocess.data_concatenate(load_k_day_ago, weather_k)

                input_vector.append(Preprocess.data_concatenate(feature, label))
            return np.array(input_vector)

        @staticmethod
        def filter_load_weather_weekly(load_vector, weather_vector):
            """
            set input label and feature vector as following
                label = load(k)
                feature = load(k-7), weather(k)
            :param load_vector: 
            :param weather_vector: 
            :return: 
                constructed input_vector
            """
            input_vector = list()
            for row in xrange(7, len(load_vector)):
                load_k = load_vector[row]
                load_k_week_ago = load_vector[row - 7]
                weather_k = weather_vector[row]

                feature = load_k
                label = Preprocess.data_concatenate(load_k_week_ago, weather_k)

                input_vector.append(Preprocess.data_concatenate(feature, label))
            return np.array(input_vector)

        @staticmethod
        def filter_load_weather_daily_weekly(load_vector, weather_vector):
            """
            set input label and feature vector as following
                label = load(k)
                feature = load(k-7), load(k-1), weather(k)
            :param load_vector: 
            :param weather_vector: 
            :return: 
                constructed input_vector
            """
            input_vector = list()
            for row in xrange(7, len(load_vector)):
                load_k = load_vector[row]
                load_k_day_ago = load_vector[row - 1]
                load_k_week_ago = load_vector[row - 7]
                weather_k = weather_vector[row]

                feature = load_k
                label = Preprocess.data_concatenate(load_k_week_ago,
                                                    Preprocess.data_concatenate(load_k_day_ago, weather_k))

                input_vector.append(Preprocess.data_concatenate(feature, label))
            return np.array(input_vector)

        @staticmethod
        def filter_load_weather_daily_weekly_powered(load_vector, weather_vector):
            """
            set input label and feature vector as following
                label = load(k)
                feature = load(k-7)^weather(k), load(k-1)^weather(k)
            :param load_vector: 
            :param weather_vector: 
            :return:
                constructed input_vector
            """
            input_vector = list()
            for row in xrange(7, len(load_vector)):
                load_k = load_vector[row]
                load_k_day_ago = load_vector[row - 1]
                load_k_week_ago = load_vector[row - 7]
                weather_k = weather_vector[row]

                load_k_day_ago_powered = Preprocess.power_function(load_k_day_ago, weather_k)
                load_k_week_ago_powered = Preprocess.power_function(load_k_week_ago, weather_k)

                feature = load_k
                label = Preprocess.data_concatenate(load_k_week_ago_powered, load_k_day_ago_powered)

                input_vector.append(Preprocess.data_concatenate(feature, label))
            return np.array(input_vector)


class SaveData:
    def __init__(self):
        pass

    class STLF:
        def __init__(self):
            pass

        @staticmethod
        def elm_based(input_vector, area_name, data_type, filter_type):
            """
            save a given input vector
            :param input_vector: 
            :param area_name: 
            :param data_type: 
            :param filter_type: 
            :return: 
            """
            save_path = SAVE_PATH
            save_path = os.path.join(save_path, 'STLF')
            save_path = os.path.join(save_path, area_name)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            file_name = data_type + '_' + filter_type + '.csv'
            file_name = os.path.join(save_path, file_name)
            with open(file_name, 'w') as csv_file:
                np.savetxt(csv_file, np.array(input_vector), delimiter=",")
            return file_name

        @staticmethod
        def input_vector(load_dict, weather_vector):
            """
            
            :param load_dict: 
            :param weather_vector: 
            :return: 
            """
            area_names = sorted(load_dict.iterkeys())
            filter_list = [WEATHER_DAILY, LOAD_WEATHER_DAILY, LOAD_WEATHER_WEEKLY, LOAD_WEATHER_DAILY_WEEKLY]
            print 'saved files list :'
            for file_idx in xrange(0, len(area_names)):
                area = area_names[file_idx]
                for filter_idx in xrange(0, len(filter_list)):
                    train_input_vector, test_input_vector = DataAllocate.STLF.allocator(load_dict[area], weather_vector,
                                                                                        filter_list[filter_idx])
                    print ' ',
                    print SaveData.STLF.elm_based(train_input_vector, area, 'train', filter_list[filter_idx])
                    print ' ',
                    print SaveData.STLF.elm_based(test_input_vector, area, 'test', filter_list[filter_idx])

        @staticmethod
        def all_in_one():
            pass


if __name__ == '__main__':
    pass
