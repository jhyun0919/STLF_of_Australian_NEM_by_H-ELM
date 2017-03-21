import cPickle as pickle

file = '/Users/JH/Documents/GitHub/PowerForecast/dataset.pkl'

with open(file, 'rb') as input:
    dataset = pickle.load(input)

    print dataset