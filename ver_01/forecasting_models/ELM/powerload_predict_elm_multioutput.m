%TrainingData_File='elm_train00_data.csv'; TestingData_File='elm_test00_data.csv';
%TrainingData_File='elm_train01_data.csv'; TestingData_File='elm_test01_data.csv';
%TrainingData_File='elm_train02_data.csv'; TestingData_File='elm_test02_data.csv';
%TrainingData_File='elm_train03_data.csv'; TestingData_File='elm_test03_data.csv';
TrainingData_File='elm_train04_data.csv'; TestingData_File='elm_test04_data.csv';

No_of_Output=48;
NumberofHiddenNeurons=50;
ActivationFunction='sig';
%rand('state',78309924);

ELM_MultiOutputRegression(TrainingData_File, TestingData_File, No_of_Output, NumberofHiddenNeurons, ActivationFunction);