# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from sklearn import preprocessing
from datetime import date
from datetime import timedelta

load_files = ['/Users/JH/Documents/GitHub/PowerForecast/ver_03/data/load/NSW_nslp2006.csv',
              '/Users/JH/Documents/GitHub/PowerForecast/ver_03/data/load/NSW_nslp2007.csv',
              '/Users/JH/Documents/GitHub/PowerForecast/ver_03/data/load/NSW_nslp2008.csv',
              '/Users/JH/Documents/GitHub/PowerForecast/ver_03/data/load/NSW_nslp2009.csv',
              '/Users/JH/Documents/GitHub/PowerForecast/ver_03/data/load/NSW_nslp2010.csv',
              '/Users/JH/Documents/GitHub/PowerForecast/ver_03/data/load/NSW_nslp2011.csv',
              '/Users/JH/Documents/GitHub/PowerForecast/ver_03/data/load/NSW_nslp2012.csv',
              '/Users/JH/Documents/GitHub/PowerForecast/ver_03/data/load/NSW_nslp2013.csv',
              '/Users/JH/Documents/GitHub/PowerForecast/ver_03/data/load/NSW_nslp2014.csv',
              '/Users/JH/Documents/GitHub/PowerForecast/ver_03/data/load/NSW_nslp2015.csv']

load_areas = ['ACTEWAGL', 'CITIPOWER', 'COUNTRYENERGY', 'ENERGYAUST', 'INTEGRAL', 'POWERCOR', 'TXU', 'UMPLP', 'UNITED',
              'VICAGL']

temperature_max_file = '/Users/JH/Documents/GitHub/PowerForecast/ver_03/data/weather/NSW_temp_max_2006_2015.csv'
temperature_min_file = '/Users/JH/Documents/GitHub/PowerForecast/ver_03/data/weather/NSW_temp_min_2006_2015.csv'
rainfall_file = '/Users/JH/Documents/GitHub/PowerForecast/ver_03/data/weather/NSW_rainfall_2006_2015.csv'
solar_expose_file = '/Users/JH/Documents/GitHub/PowerForecast/ver_03/data/weather/NSW_solar_2006_2015.csv'


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


class ReadCSV:
    def __init__(self):
        pass

    @staticmethod
    def load(csv_files=load_files):
        load_data = dict()

        for file_num in xrange(0, len(csv_files)):
            df = pd.read_csv(csv_files[file_num])
            for area_num in xrange(0, len(load_areas)):
                temp_load = (df.loc[df['ProfileArea'] == load_areas[area_num]]).ix[:, 4:52]
                if file_num == 0:
                    load_data[load_areas[area_num]] = np.array(temp_load)
                else:
                    load_data[load_areas[area_num]] = np.vstack((load_data[load_areas[area_num]], temp_load))

        return load_data

    @staticmethod
    def weather():
        """
        build train and test vector of weather data
        :return: 
        """
        return Preprocess.integrate_weather_data(ReadCSV.temp_max(), ReadCSV.temp_min(), ReadCSV.rainfall(),
                                                 ReadCSV.solar())

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


class Numpy2Pandas:
    def __init__(self):
        pass

    @staticmethod
    def set_date_list(start_date, date_list_length):
        date_list = list()
        present_date = start_date
        for idx in xrange(0, date_list_length):
            date_list.append(present_date)
            present_date = present_date + timedelta(days=1)
        return np.array(date_list)

    @staticmethod
    def add_date_index(data):
        start_date = date(2006, 1, 1)
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
    def load_data2df(load_data):
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
        data_dictionary = dict()
        weather_key = ['Temperature_Max', 'Temperature_min', 'Rain_fall', 'Solar_exposure']
        for idx in xrange(0, 4):
            data_dictionary[weather_key[idx]] = weather_data[:, idx]
        data_dictionary = pd.DataFrame(data_dictionary)

        return Numpy2Pandas.add_date_index(data_dictionary)


class DataAllocate:
    def __init__(self):
        pass


class SaveData:
    def __init__(self):
        pass

    @staticmethod
    def elm_based(data_vector, data_type, combine_type, filter_type=None):
        if filter_type is None:
            with open('elm_' + data_type + '_' + combine_type + '_data.csv', 'w') as csv_file:
                np.savetxt(csv_file, np.array(data_vector), delimiter=",")


if __name__ == '__main__':
    pass
