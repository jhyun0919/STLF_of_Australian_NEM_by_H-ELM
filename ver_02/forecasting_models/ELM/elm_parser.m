function [ ] = elm_parser( TrainingData_File, TestingData_File , No_of_Output, FileName)

%%%%%%%%%%% Load training dataset
train_data=csvread(TrainingData_File);
T=train_data(:,1:No_of_Output)';
P=train_data(:,No_of_Output+1:size(train_data,2))';

train_x=P;
train_y=T;

clear train_data;                                   %   Release raw training data array

%%%%%%%%%%% Load testing dataset
test_data=csvread(TestingData_File);
TV.T=test_data(:,1:No_of_Output)';
TV.P=test_data(:,No_of_Output+1:size(test_data,2))';

test_x=TV.P;
test_y=TV.T;

clear test_data;                                    %   Release raw testing data array



save(FileName, train_x, train_y, test_x, test_y);


end

