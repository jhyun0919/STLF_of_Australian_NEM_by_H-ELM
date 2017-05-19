%load elm_input_data00.mat; Result_File='elm_forecast_result00.csv'; NumberofHiddenNeurons=25;
%load elm_input_data01.mat; Result_File='elm_forecast_result01.csv'; NumberofHiddenNeurons=40;
%load elm_input_data02.mat; Result_File='elm_forecast_result02.csv'; NumberofHiddenNeurons=30;
load elm_input_data03.mat; Result_File='elm_forecast_result03.csv'; NumberofHiddenNeurons=30;

No_of_Output=48;
ActivationFunction='sig';

AccList = [];

for x = 1:1000
x
[TrainingTime, TestingTime, TrainingAccuracy, TestingAccuracy] = ELM_MultiOutputRegression(train_x, train_y, test_x, test_y, No_of_Output, NumberofHiddenNeurons, ActivationFunction);
AccList(x,:) = TestingAccuracy;

end

headers = {'MAPE'};

csvwrite_with_headers(Result_File,AccList,headers)