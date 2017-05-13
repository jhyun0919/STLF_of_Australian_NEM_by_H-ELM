clear;clc;
%% This is the demo for MNIST dataset.
rand('state',0)
%% Dataset
% We used the original MNIST dataset, the overall samples are simply scaled to [0,1]. 
% And the labels are resized to 10 dimension which are -1 or 1.
load mnist.mat;
%% Randomness
% Due to the random initialization of neural networks, it is very common to see that the performance has fluctuations. 
% To reproduce the exact same testing results in our paper, please use the attached .mat file obtained by following codes:

% rand('state',16917921)   % 5000
% rand('state',67797325)   % 12000
% b1=2*rand(size(train_x,2)+1,N1)-1;
% b2=2*rand(N1+1,N2)-1;
% b3=orth(2*rand(N2+1,N3)'-1)';

% And our testing hardware and software conditions are as follows:
% Laptop, Intel-i7 2.4G CPU, 16G DDR3 RAM, Windows 7, Matlab R2013b. (We have also tested the codes on Matlab 2014b)

% If you have further interests, you could try to build random mapping matrices, add other preprocessing or tuning tricks by you own, :-).
%% To achieve the 99.13% accuracy of H-ELM as we did in paper, load the random matrices which are totally independent from the training data.
C = 2^-30; s = .8; % C is the L2 penalty of the last layer ELM and s is the scaling factor.
load random_700_700_12000.mat; 
%% If the RAM of your computer is less than 16G, you may try the following version which has less than half of the hidden nodes
% C = 2^-30 ;s = 1;
% load random_700_700_5000.mat;
%% Call the training function
[TrainingAccuracy, TestingAccuracy, Training_time, Testing_time] = helm_train(train_x, train_y, test_x, test_y, b1, b2, b3, s, C);
