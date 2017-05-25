load 'input_data/ACTEWAGL/helm_input_wd.mat';Result_File='forecast_result/ACTEWAGL/layer1/helm_forecast_result_wd.csv'; NumberofHiddenNeurons=52;
%load 'input_data/ACTEWAGL/helm_input_lwd.mat'; Result_File='forecast_result/ACTEWAGL/layer1/helm_forecast_result_lwd.csv'; NumberofHiddenNeurons=129; 
%load 'input_data/ACTEWAGL/helm_input_lww.mat'; Result_File='forecast_result/ACTEWAGL/layer1/helm_forecast_result_lww.csv'; NumberofHiddenNeurons=129; 
%load 'input_data/ACTEWAGL/helm_input_lwdw.mat'; Result_File='forecast_result/ACTEWAGL/layer1/helm_forecast_result_lwdw.csv'; NumberofHiddenNeurons=170; 
 



N1=NumberofHiddenNeurons;
N=N1+1;

for x = 1:1000
x
b1=2*rand(size(train_x',2)+1,N1)-1;
b=orth(2*rand(N1+1,N)'-1)';
C = 2^-30; s = .8;

fprintf(1,'N1= %d\n',N1);
fprintf(1,'N= %d\n',N);

[TrainingAccuracy_RMSE,TestingAccuracy_RMSE,Training_time,Testing_time] = helm_regression_01(train_x, train_y, test_x, test_y, b1, b, s, C);
RMSE_Training(x,:) = TrainingAccuracy_RMSE;
RMSE_Testing(x,:) = TestingAccuracy_RMSE;
end

AccList = horzcat(RMSE_Training, RMSE_Testing);
headers = {'Training', 'Testing'};

csvwrite_with_headers(Result_File,AccList,headers)