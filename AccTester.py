# -*- coding: utf-8 -*-

import numpy as np
import cmath as math

pl_target = np.array([0.31445369, 0.26013332, 0.2690349, 0.2598849, 0.18246181, 0.11986089,
                      0.06015816, 0.03924978, 0., 0.01167557, 0.02865069, 0.13973419,
                      0.2552478, 0.46222001, 0.67105536, 0.88312011, 0.98538484, 0.96990022,
                      0.9407527, 0.92282532, 0.87935246, 0.85554589, 0.80577982, 0.77630108,
                      0.76408728, 0.73514677, 0.70496419, 0.70260423, 0.68861011, 0.6640583,
                      0.65242413, 0.64360535, 0.67941871, 0.73291103, 0.82159566, 0.87152735,
                      0.97524117, 0.97076968, 1., 0.98066493, 0.99374819, 0.98575746,
                      0.91437917, 0.8358796, 0.78006873, 0.71320333, 0.60820602, 0.5380698])

pl_forcasted = np.array([0.53542956, 0.45387472, 0.34691755, 0.22470909, 0.13235949, 0.06645209,
                         0.03535529, 0.02342164, 0.00361476, 0., 0.0031691, 0.07036395,
                         0.13186432, 0.23698935, 0.33384501, 0.5177024, 0.65550879, 0.78252043,
                         0.81648923, 0.84407031, 0.79014608, 0.74384749, 0.6857638, 0.6258975,
                         0.58276801, 0.53230998, 0.49140876, 0.46239168, 0.44416935, 0.43619708,
                         0.405744, 0.43312701, 0.43892052, 0.48214905, 0.58554098, 0.75533548,
                         0.93018074, 0.98039119, 1., 0.98702649, 0.99341421, 0.97974746,
                         0.93978708, 0.86764051, 0.78153008, 0.69373607, 0.63585046, 0.57766774])


def rmse(target_data_vector, forecasted_data_vector):
    """
    root mean square error
    :param target_data_vector:
    :param forecasted_data_vector:
    :return:
    """
    return np.sqrt(np.mean((forecasted_data_vector - target_data_vector) ** 2))


def MAPE(target_data_vector, forecasted_data_vector):
    """

    :param target_data_vector:
    :param forecasted_data_vector:
    :return:
    """
    nominator = abs(target_data_vector - forecasted_data_vector)
    denominator = abs(target_data_vector)
    coefficient = 1 / len(target_data_vector)
    print nominator
    print denominator
    error_distance = nominator / denominator

    print sum(error_distance)


def MAE(target_data_vector, forecasted_data_vector):
    """

    :param original_data_vector:
    :param forecasted_data_vector:
    :return:
    """
    return (1 / len(target_data_vector)) * sum(abs(target_data_vector - forecasted_data_vector))


def euclidean_distance(data_vector):
    """
    calculate euclidean distance from the start and end of a given vector
    :param vector:

    :return:
        euclidean distance
    """
    squared_sum = 0
    for i in xrange(0, len(data_vector)):
        squared_sum = data_vector[i] * data_vector[i] + squared_sum

    return math.sqrt(squared_sum)


if __name__ == '__main__':
    # print pl_o
    # print type(pl_o)
    # print pl_o.shape

    # print pl_p
    # print type(pl_p)
    # print pl_p.shape

    # print 'MAPE : ',
    # print MAPE(pl_p, pl_o)

    # print 'MAE : ',
    # print MAE(pl_p, pl_o)
    print rmse(pl_target, pl_forcasted)
