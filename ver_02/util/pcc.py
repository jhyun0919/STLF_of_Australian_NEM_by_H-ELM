# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import math

load_file = '/Users/JH/Documents/GitHub/PowerForecast/ver_01/data/NSW/load/NSW_NSLP_VICAGL_2006_2015.csv'
day_type = 1
week_type = 7


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


class PCC:
    def __init__(self):
        pass

    @staticmethod
    def point(x, y):
        """
        Pearson's correlation coefficient (PCC)
        :param x: 
        :param y: 
        :return: 
        """
        numerator = list()
        denominator_x = list()
        denominator_y = list()
        mean_x = np.mean(x)
        mean_y = np.mean(y)
        for idx in xrange(0, len(x)):
            numerator.append((x[idx] - mean_x) * (y[idx] - mean_y))
            denominator_x.append(math.pow((x[idx] - mean_x), 2))
            denominator_y.append(math.pow((y[idx] - mean_y), 2))

        numerator = np.sum(np.array(numerator))
        denominator = math.sqrt(np.sum(np.array(denominator_x))) * math.sqrt(np.sum(np.array(denominator_y)))

        return numerator / denominator

    @staticmethod
    def average(group_type, index_diff):
        data = ReadCSV.laod()
        pcc_score = list()

        for row in xrange(0, len(data) - group_type * index_diff):
            pcc_score.append(PCC.point(data[row], data[row + group_type * index_diff]))

        return np.mean(np.array(pcc_score))

    @staticmethod
    def list(group_type):
        pcc_list = list()
        for idx in xrange(1, 11):
            pcc_list.append(PCC.average(group_type, idx))
        return np.array(pcc_list)


if __name__ == '__main__':
    print PCC.list(day_type)
    print PCC.list(week_type)
