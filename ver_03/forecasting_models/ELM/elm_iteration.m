%load 'input_data/elm_input_d.mat'; Result_File='forecast_result/elm_forecast_result00.csv'; NumberofHiddenNeurons=80; 
%load 'input_data/elm_input_w.mat'; Result_File='forecast_result/elm_forecast_result01.csv'; NumberofHiddenNeurons=120; 
load 'input_data/elm_input_dw.mat'; Result_File='forecast_result/elm_forecast_result02.csv'; NumberofHiddenNeurons=120; 


No_of_Output=48;
ActivationFunction='sig';

RMSE_List = [];

for x = 1:1000
x
[TrainingTime, TestingTime, TrainingAccuracy_RMSE, TestingAccuracy_RMSE] = ELM_MultiOutputRegression(train_x, train_y, test_x, test_y, No_of_Output, NumberofHiddenNeurons, ActivationFunction);
RMSE_List(x,:) = TestingAccuracy_RMSE;
end

AccList = horzcat(RMSE_List);
headers = {'RMSE'};
%csvwrite_with_headers(Result_File,AccList,headers)


clear;

