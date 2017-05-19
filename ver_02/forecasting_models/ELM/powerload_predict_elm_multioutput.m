%load elm_input_data00.mat; Result_File='elm_forecast_result00.csv';
%load elm_input_data01.mat; Result_File='elm_forecast_result01.csv';
%load elm_input_data02.mat; Result_File='elm_forecast_result02.csv';
load elm_input_data03.mat; Result_File='elm_forecast_result03.csv';

No_of_Output=48;
NumberofHiddenNeurons=50;
ActivationFunction='sig';
%rand('state',78309924);

ELM_MultiOutputRegression(train_x, train_y, test_x, test_y, No_of_Output, NumberofHiddenNeurons, ActivationFunction);