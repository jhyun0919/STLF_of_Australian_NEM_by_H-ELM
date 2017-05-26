load 'input_data/ACTEWAGL/elm_input_wd.mat'; Result_File='forecast_result/ACTEWAGL/elm_forecast_result_wd.csv'; NumberofHiddenNeurons=52; 
%load 'input_data/ACTEWAGL/elm_input_lwd.mat'; Result_File='forecast_result/ACTEWAGL/elm_forecast_result_lwd.csv'; NumberofHiddenNeurons=129; 
%load 'input_data/ACTEWAGL/elm_input_lww.mat'; Result_File='forecast_result/ACTEWAGL/elm_forecast_result_lww.csv'; NumberofHiddenNeurons=130; 
%load 'input_data/ACTEWAGL/elm_input_lwdw.mat'; Result_File='forecast_result/ACTEWAGL/elm_forecast_result_lwdw.csv'; NumberofHiddenNeurons=170; 




No_of_Output=48;
ActivationFunction='sig';


for x = 1:1000
x
[TrainingTime, TestingTime, TrainingAccuracy_RMSE, TestingAccuracy_RMSE] = ELM_MultiOutputRegression(train_x, train_y, test_x, test_y, No_of_Output, NumberofHiddenNeurons, ActivationFunction);
TrainingTimeList(x,:) = TrainingTime;
RMSE_Training(x,:) = TrainingAccuracy_RMSE;
RMSE_Testing(x,:) = TestingAccuracy_RMSE;
end

AccList = horzcat(RMSE_Training, RMSE_Testing);
headers = {'Training', 'Testing'};
csvwrite_with_headers(Result_File,AccList,headers);
TrainingtTime = mean(TrainingTimeList)