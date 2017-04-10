# -*- coding: utf-8 -*-

import numpy as np
import scipy.io as sio
import cPickle as pickle


file_directory = '/Users/JH/Documents/GitHub/PowerForecast/ELM_based/elm_output.mat'
mat_contents = sio.loadmat(file_directory)
print mat_contents['output']
print mat_contents['output'].shape

data = np.transpose(mat_contents['output'])
print data
print data.shape

f = open('elm_ouput.txt', 'w')
pickle.dump(data, f)
f.close()