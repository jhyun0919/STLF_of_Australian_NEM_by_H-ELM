function [ TrainingTime, TrainingAccuracy ] = elm_multioutput_regression_train( TrainingData_File, No_of_Output, NumberofHiddenNeurons, ActivationFunction )

% Usage: elm_multiouput_regression_train( TrainingData_File, No_of_Output, NumberofHiddenNeurons, ActivationFunction )
% OR:    [TrainingTime, TrainingAccuracy] = elm_multiouput_regression_train( TrainingData_File, No_of_Output, NumberofHiddenNeurons, ActivationFunction )
%
% Input:
% TrainingData_File     - Filename of training data set
% No_of_Output          - Number of outputs for regression
% NumberofHiddenNeurons - Number of hidden neurons assigned to the ELM
% ActivationFunction    - Type of activation function:
%                           'sig' for Sigmoidal function
%                           'sin' for Sine function
%                           'hardlim' for Hardlim function
%
% Output: 
% TrainingTime          - Time (seconds) spent on training ELM
% TrainingAccuracy      - Training accuracy: 
%                           RMSE for regression

%
    %%%%    Authors:    MR QIN-YU ZHU AND DR GUANG-BIN HUANG
    %%%%    NANYANG TECHNOLOGICAL UNIVERSITY, SINGAPORE
    %%%%    EMAIL:      EGBHUANG@NTU.EDU.SG; GBHUANG@IEEE.ORG
    %%%%    WEBSITE:    http://www.ntu.edu.sg/eee/icis/cv/egbhuang.htm
    %%%%    DATE:       APRIL 2004
    
    %%%%    fixed by Jee Hyun Park (jhyun19@gmail.com), MAR 2017

%%%%%%%%%%% Load training dataset
train_data=csvread(TrainingData_File);
T=train_data(:,1:No_of_Output)';
P=train_data(:,No_of_Output+1:size(train_data,2))';
clear train_data;                                   %   Release raw training data array

NumberofTrainingData=size(P,2);
NumberofInputNeurons=size(P,1);

%%%%%%%%%%% Calculate weights & biases
start_time_train=cputime;

%%%%%%%%%%% Random generate input weights InputWeight (w_i) and biases BiasofHiddenNeurons (b_i) of hidden neurons
InputWeight=rand(NumberofHiddenNeurons,NumberofInputNeurons)*2-1;
BiasofHiddenNeurons=rand(NumberofHiddenNeurons,1);
tempH=InputWeight*P;
clear P;                                            %   Release input of training data 
ind=ones(1,NumberofTrainingData);
BiasMatrix=BiasofHiddenNeurons(:,ind);              %   Extend the bias matrix BiasofHiddenNeurons to match the demention of H
tempH=tempH+BiasMatrix;

%%%%%%%%%%% Calculate hidden neuron output matrix H
switch lower(ActivationFunction)
    case {'sig','sigmoid'}
        %%%%%%%% Sigmoid 
        H = 1 ./ (1 + exp(-tempH));
    case {'sin','sine'}
        %%%%%%%% Sine
        H = sin(tempH);    
    case {'hardlim'}
        %%%%%%%% Hard Limit
        H = hardlim(tempH);            
        %%%%%%%% More activation functions can be added here
    case {'relu'}
        %%%%%%%% RelU (still need to be checked... not working currently...)
        H = relu(tempH);                 
end
clear tempH;                                        %   Release the temparary array for calculation of hidden neuron output matrix H

%%%%%%%%%%% Calculate output weights OutputWeight (beta_i)
OutputWeight=pinv(H') * T';
end_time_train=cputime;
TrainingTime=end_time_train-start_time_train        %   Calculate CPU time (seconds) spent for training ELM

%%%%%%%%%%% Calculate the training accuracy
Y=(H' * OutputWeight)';                             %   Y: the actual output of the training data
TrainingAccuracy=sqrt(mse(T - Y))               %   Calculate training accuracy (RMSE) for regression case
clear H;

%%%%%%%%%%% Save the model
save('elm_model', 'No_of_Output', 'NumberofInputNeurons', 'NumberofHiddenNeurons', 'InputWeight', 'BiasofHiddenNeurons', 'OutputWeight', 'ActivationFunction');
    
end

