# -*- coding: utf-8 -*-

import tensorflow as tf
import pandas as pd
from data_parser import data_alloter

file_directory = '/Users/JH/Desktop/NTU/NTU_Research/data/NEM_Load_Forecasting_Database.xls'
logs_path = './tensorflow_logs/5_layers'

QLD = 'Actual_Data_QLD'
NSW = 'Actual_Data_NSW'
VIC = 'Actual_Data_VIC'
SA = 'Actual_Data_SA'
TAS = 'Actual_Data_TAS'

df = pd.read_excel(file_directory, sheetname=QLD)
data_set = data_alloter(df)

CONSTANT = tf.app.flags


# Parameters
CONSTANT.DEFINE_integer('num_steps', 3000, 'number of iterations')
CONSTANT.DEFINE_integer('data_showing_step', 100, 'data showing step')
CONSTANT.DEFINE_integer('batch_size', 30, 'size of the batch')

CONSTANT.DEFINE_integer('samples', len(df), 'data length')
CONSTANT.DEFINE_integer('state_size', 144, )


# Network Parameter
CONSTANT.DEFINE_integer('n_hidden', 80, 'number of perceptron in a layer')
CONSTANT.DEFINE_integer('n_input', 144, 'number of input')
CONSTANT.DEFINE_integer('n_output', 48, 'number of output')





class SingleLayer(object):
    def __init__(self):
        self._initialize()

    def run(self):
        pass


    @classmethod
    def _gen_data(cls):
        ts_x = data_set.PreProcessed.Train.feature
        ts_y = data_set.PreProcessed.Train.target


    @classmethod
    def _initialize(cls):
        cls.sess = tf.Session()



    @classmethod
    def _run_session(cls):
        pass




def main():
    pass


if __name__== "__main__":
    tf.app.run()