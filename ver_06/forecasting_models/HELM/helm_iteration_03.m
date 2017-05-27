load 'input_data/ACTEWAGL/helm_input_wd.mat';Result_File='forecast_result/ACTEWAGL/layer3/helm_forecast_result_wd.csv'; NumberofHiddenNeurons=52;
%load 'input_data/ACTEWAGL/helm_input_lwd.mat'; Result_File='forecast_result/ACTEWAGL/layer3/helm_forecast_result_lwd.csv'; NumberofHiddenNeurons=129; 
%load 'input_data/ACTEWAGL/helm_input_lww.mat'; Result_File='forecast_result/ACTEWAGL/layer3/helm_forecast_result_lww.csv'; NumberofHiddenNeurons=129; 
%load 'input_data/ACTEWAGL/helm_input_lwdw.mat'; Result_File='forecast_result/ACTEWAGL/layer3/helm_forecast_result_lwdw.csv'; NumberofHiddenNeurons=170; 



N1=NumberofHiddenNeurons;
N2=50;
N3=50;
N=N3+1;

RMSE_List = [];

for x = 1:1000
x
b1=2*rand(size(train_x',2)+1,N1)-1;
b2=2*rand(N1+1,N2)-1;
b3=2*rand(N2+1,N3)-1;
b=orth(2*rand(N3+1,N)'-1)';
C = 2^-30; s = .8;

fprintf(1,'N1= %d\n',N1);
fprintf(1,'N2= %d\n',N2);
fprintf(1,'N3= %d\n',N3);
fprintf(1,'N= %d\n',N);

[TrainingAccuracy,TestingAccuracy_RMSE,Training_time,Testing_time] = helm_regression_03(train_x, train_y, test_x, test_y, b1, b2, b3, b, s, C);
RMSE_List(x,:) = TestingAccuracy_RMSE;
end

AccList = horzcat(RMSE_List);
headers = {'RMSE'};
csvwrite_with_headers(Result_File,AccList,headers)