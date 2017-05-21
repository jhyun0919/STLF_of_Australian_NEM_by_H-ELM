# -*- coding: utf-8 -*-

import numpy as np
import math

day_type = 1
week_type = 7


def get_point(x, y):
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


def get_average(data, group_type, index_diff):
    pcc_score = list()

    for row in xrange(0, len(data) - group_type * index_diff):
        pcc_score.append(get_point(data[row], data[row + group_type * index_diff]))

    return np.mean(np.array(pcc_score))


def get_list(data, group_type, list_len=10):
    pcc_list = list()
    for idx in xrange(0, list_len):
        pcc_list.append(get_average(data, group_type, idx))
    return np.array(pcc_list)


if __name__ == '__main__':
    pass
