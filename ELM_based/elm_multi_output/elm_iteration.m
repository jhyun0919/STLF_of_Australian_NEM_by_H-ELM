%TrainingData_File='Actual_Data_QLD_preprocessed_train.csv'; TestingData_File='Actual_Data_QLD_preprocessed_test.csv';
TrainingData_File='Actual_Data_QLD_raw_train.csv'; TestingData_File='Actual_Data_QLD_raw_test.csv';

No_of_Output=48;
NumberofHiddenNeurons=50;
ActivationFunction='sig';

AccList = [];

for x = 1:1000

[TrainingTime, TestingTime, TrainingAccuracy, TestingAccuracy] = ELM_MultiOutputRegression(TrainingData_File, TestingData_File, No_of_Output, NumberofHiddenNeurons, ActivationFunction);
AccList(x,:) = TestingAccuracy;

end

headers = {'RMSE'};

%csvwrite_with_headers('elm_preprocessed_result.csv',AccList,headers)
csvwrite_with_headers('elm_raw_result.csv',AccList,headers)