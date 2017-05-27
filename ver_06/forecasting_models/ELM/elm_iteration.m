%load 'input_data/ACTEWAGL/elm_input_wd.mat'; Result_File='forecast_result/ACTEWAGL/elm_forecast_result_wd.csv'; NumberofHiddenNeurons=52; 
%load 'input_data/ACTEWAGL/elm_input_lwd.mat'; Result_File='forecast_result/ACTEWAGL/elm_forecast_result_lwd.csv'; NumberofHiddenNeurons=178; 
%load 'input_data/ACTEWAGL/elm_input_lww.mat'; Result_File='forecast_result/ACTEWAGL/elm_forecast_result_lww.csv'; NumberofHiddenNeurons=148; 
load 'input_data/ACTEWAGL/elm_input_lwdw.mat'; Result_File='forecast_result/ACTEWAGL/elm_forecast_result_lwdw.csv'; NumberofHiddenNeurons=167; 




No_of_Output=48;
ActivationFunction='sig';

RMSE_Training = [];
RMSE_Testing = [];
MAPE_Training = [];
MAPE_Testing = [];
MAE_Training = [];
MAE_Testing = [];

for x = 1:1000
x
[TrainingTime, TestingTime, TrainingAccuracy_RMSE, TestingAccuracy_RMSE, TrainingAccuracy_MAPE, TestingAccuracy_MAPE, TrainingAccuracy_MAE, TestingAccuracy_MAE] = ELM_MultiOutputRegression(train_x, train_y, test_x, test_y, No_of_Output, NumberofHiddenNeurons, ActivationFunction);
TrainingTimeList(x,:) = TrainingTime;

RMSE_Training(x,:) = TrainingAccuracy_RMSE;
RMSE_Testing(x,:) = TestingAccuracy_RMSE;
MAPE_Training(x,:) = TrainingAccuracy_MAPE;
MAPE_Testing(x,:) = TestingAccuracy_MAPE;
MAE_Training(x,:) = TrainingAccuracy_MAE;
MAE_Testing(x,:) = TestingAccuracy_MAE;
end

AccList = horzcat(RMSE_Training, RMSE_Testing, MAPE_Training, MAPE_Testing, MAE_Training, MAE_Testing);
headers = {'RMSE_Train', 'RMSE_Test', 'MAPE_Train', 'MAPE_Test', 'MAE_Train', 'MAE_Test'};
csvwrite_with_headers(Result_File,AccList,headers);
TrainingtTime = mean(TrainingTimeList)