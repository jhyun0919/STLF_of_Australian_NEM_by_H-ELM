%TrainingData_File='Actual_Data_QLD_preprocessed_train.csv'; TestingData_File='Actual_Data_QLD_preprocessed_test.csv';

TrainingData_File='Actual_Data_QLD_raw_train.csv'; TestingData_File='Actual_Data_QLD_raw_test.csv';

No_of_Output=48;
NumberofHiddenNeurons=50;
ActivationFunction='sig';
%rand('state',78309924);

ELM_MultiOutputRegression(TrainingData_File, TestingData_File, No_of_Output, NumberofHiddenNeurons, ActivationFunction);