%load 'input_data/ACTEWAGL/helm_input_wd.mat';Result_File='forecast_result/ACTEWAGL/layer2/helm_forecast_result_wd.csv'; NumberofHiddenNeurons=148;
%load 'input_data/ACTEWAGL/helm_input_lwd.mat'; Result_File='forecast_result/ACTEWAGL/layer2/helm_forecast_result_lwd.csv'; NumberofHiddenNeurons=208; 
%load 'input_data/ACTEWAGL/helm_input_lww.mat'; Result_File='forecast_result/ACTEWAGL/layer2/helm_forecast_result_lww.csv'; NumberofHiddenNeurons=198; 
%load 'input_data/ACTEWAGL/helm_input_lwdw1.mat'; Result_File='forecast_result/ACTEWAGL/layer2/helm_forecast_result_lwdw1.csv'; NumberofHiddenNeurons=159;
%load 'input_data/ACTEWAGL/helm_input_lwdw2.mat'; Result_File='forecast_result/ACTEWAGL/layer2/helm_forecast_result_lwdw2.csv'; NumberofHiddenNeurons=159;
%load 'input_data/ACTEWAGL/helm_input_lwdw3.mat'; Result_File='forecast_result/ACTEWAGL/layer2/helm_forecast_result_lwdw3.csv'; NumberofHiddenNeurons=159;
load 'input_data/ACTEWAGL/helm_input_lwdw4.mat'; Result_File='forecast_result/ACTEWAGL/layer2/helm_forecast_result_lwdw4.csv'; NumberofHiddenNeurons=159;
%load 'input_data/ACTEWAGL/helm_input_lwdw5.mat'; Result_File='forecast_result/ACTEWAGL/layer2/helm_forecast_result_lwdw5.csv'; NumberofHiddenNeurons=159;
%load 'input_data/ACTEWAGL/helm_input_lwdw6.mat'; Result_File='forecast_result/ACTEWAGL/layer2/helm_forecast_result_lwdw6.csv'; NumberofHiddenNeurons=159; 
 



N1=NumberofHiddenNeurons;
N2=NumberofHiddenNeurons;
N=N2+1;



RMSE_Training = [];
RMSE_Testing = [];
MAPE_Training = [];
MAPE_Testing = [];
MAE_Training = [];
MAE_Testing = [];

for x = 1:1000
x    
b1=2*rand(size(train_x',2)+1,N1)-1;
b2=2*rand(N1+1,N2)-1;
b=orth(2*rand(N2+1,N)'-1)';
C = 2^-30; s = .8;

fprintf(1,'N1= %d\n',N1);
fprintf(1,'N= %d\n',N);

[Training_time,Testing_time, TrainingAccuracy_RMSE, TestingAccuracy_RMSE, TrainingAccuracy_MAPE, TestingAccuracy_MAPE, TrainingAccuracy_MAE, TestingAccuracy_MAE] = helm_regression_02(train_x, train_y, test_x, test_y, b1, b2, b, s, C);
TrainingTimeList(x,:) = Training_time;

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