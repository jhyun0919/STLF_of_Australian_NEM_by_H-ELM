clear;clc;
%% This is the demo for NORB dataset.
rand('state',0)
%% Dataset
% We used the original NORB dataset, the overall samples are simply scaled to [0,1].
% And the labels are resized to 5 dimensions -1 or 1.
load norb.mat;
train_x = reshape(train_x,2048,24300)';
test_x = reshape(test_x,2048,24300)';
%% zscore and ZCA whiten
train_x = bsxfun(@rdivide, bsxfun(@minus, train_x, mean(train_x,2)), sqrt(var(train_x,[],2)+10));
 
C = cov(train_x);
M = mean(train_x);
[V,D] = eig(C);
P = V * diag(sqrt(1./(diag(D) + 1e2))) * V';
train_x = bsxfun(@minus, train_x, M) * P;
 
test_x = bsxfun(@rdivide, bsxfun(@minus, test_x, mean(test_x,2)), sqrt(var(test_x,[],2)+10));
test_x = bsxfun(@minus, test_x, M) * P;
%% Randomness
% Due to the random initialization of neural networks, it is very common to see that the performance has fluctuations. 
% To reproduce the exact same testing results in our paper, please use the attached .mat file obtained by following codes:

% rand('state',78309924)  
% b1=2*rand(size(train_x,2)+1,N1)-1;
% b2=2*rand(N1+1,N2)-1;
% b3=orth(2*rand(N2+1,N3)'-1)';

% And our testing hardware and software conditions are as follows:
% Laptop, Intel-i7 2.4G CPU, 16G DDR3 RAM, Windows 7, Matlab R2013b. (We have also tested the codes on Matlab 2014b)

% If you have further interests, you could try to build random mapping matrices, add other preprocessing or tuning tricks by you own, :-).
%% To achieve the 91.28% H-ELM we as we did in paper, load the random matrices which are totally independent from the training data.
C = 2^-30 ; s = .8;
load random_3000_3000_15000.mat;
%%
[TrainingAccuracy, TestingAccuracy, Training_time, Testing_time] = helm_train(train_x, train_y, test_x, test_y, b1, b2, b3, s, C);
