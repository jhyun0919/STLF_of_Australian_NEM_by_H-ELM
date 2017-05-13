The demo consists of two parts: source codes and data. In order to reproduce the exact results of experiments in our paper, please use the same simulation data, including testing datasets (MNIST and NORB) and random matrices, which can be provided upon request via email: cwdeng@bit.edu.cn

To use these codes, you can simply unzip all files into the same path and then run the "demo_MNIST.m" and "demo_NORB.m".

The main training function "helm_train()" could be called as follows:

Example: 

[TrainingAccuracy, TestingAccuracy, Training_time, Testing_time] = helm_train(train_x, train_y, test_x, test_y, b1, b2, b3, s, C);

% train_x is the training data and train_y is the training label.
% test_x is the training data and test_y is the training label.
% b1, b2 and b3 are the random matrices, they are pre-stored in our demo and can be used by loading the random*.mat.
% C is the L2 penalty of the last layer ELM and s is the scaling factor of the activation function.	

Please cite our paper in your publications if it helps your research:

@article{tang2015helm, 
author={Tang, Jiexiong and Deng, Chenwei and Huang, Guang-Bin.}, 
journal={IEEE Transactions on Neural Networks and Learning Systems}, 
title={Extreme Learning Machine for Multilayer Perceptron}, 
year={2015},
doi={10.1109/TNNLS.2015.2424995}, 
ISSN={2162-237X},}

Enjoy, :-).