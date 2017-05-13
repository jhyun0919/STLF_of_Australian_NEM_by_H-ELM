function [ TestingTime, TestingAccuracy ] = elm_multioutput_regression_predict( TestingData_File )
% Usage: elm_multiouput_regression_predict( TestingData_File )
% OR:    [TestingTime, TestingAccuracy] = elm_multiouput_regression_train( TestingData_File )
%
% Input:
% TestingData_File     - Filename of testing data set
%
% Output: 
% TestingTime          - Time (seconds) spent on testing ELM
% TestingAccuracy      - Testing accuracy: 
%                           RMSE for regression

%
    %%%%    Authors:    MR QIN-YU ZHU AND DR GUANG-BIN HUANG
    %%%%    NANYANG TECHNOLOGICAL UNIVERSITY, SINGAPORE
    %%%%    EMAIL:      EGBHUANG@NTU.EDU.SG; GBHUANG@IEEE.ORG
    %%%%    WEBSITE:    http://www.ntu.edu.sg/eee/icis/cv/egbhuang.htm
    %%%%    DATE:       APRIL 2004
    
    %%%%    fixed by Jee Hyun Park (jhyun19@gmail.com), MAR 2017

    
    
%%%%%%%%%%% Load testing dataset

load elm_model.mat;

test_data=csvread(TestingData_File);
TV.T=test_data(:,1:No_of_Output)';
TV.P=test_data(:,No_of_Output+1:size(test_data,2))';
clear test_data;      

NumberofTestingData=size(TV.P,2);
%NumberofInputNeurons=size(TV.P,1);


%%%%%%%%%%% Calculate the output of testing input
start_time_test=cputime;
tempH_test=InputWeight*TV.P;
clear TV.P;             %   Release input of testing data             
ind=ones(1,NumberofTestingData);
BiasMatrix=BiasofHiddenNeurons(:,ind);              %   Extend the bias matrix BiasofHiddenNeurons to match the demention of H
tempH_test=tempH_test + BiasMatrix;
switch lower(ActivationFunction)
    case {'sig','sigmoid'}
        %%%%%%%% Sigmoid 
        H_test = 1 ./ (1 + exp(-tempH_test));
    case {'sin','sine'}
        %%%%%%%% Sine
        H_test = sin(tempH_test);        
    case {'hardlim'}
        %%%%%%%% Hard Limit
        H_test = hardlim(tempH_test);        
        %%%%%%%% More activation functions can be added here        
end
TY=(H_test' * OutputWeight)';                       %   TY: the actual output of the testing data
output = TY
end_time_test=cputime;
TestingTime=end_time_test-start_time_test           %   Calculate CPU time (seconds) spent by ELM predicting the whole testing data

TestingAccuracy=sqrt(mse(TV.T - TY))            %   Calculate testing accuracy (RMSE) for regression case
    
save('elm_predict_result','output');
    
    
    
end

