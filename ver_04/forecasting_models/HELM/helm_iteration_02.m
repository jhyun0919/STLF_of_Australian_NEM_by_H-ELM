%load 'input_data/helm_input_d.mat'; Result_File='forecast_result/helm_02_forecast_result00.csv';
%load 'input_data/helm_input_w.mat'; Result_File='forecast_result/helm_02_forecast_result01.csv'; 
%load 'input_data/helm_input_dw.mat'; Result_File='forecast_result/helm_02_forecast_result02.csv';
load 'input_data/helm_input.mat'; Result_File='forecast_result/helm_02_forecast_result.csv';

N1=50;
N2=50;
N=N2+1;

RMSE_List = [];


for x = 1:10
    
b1=2*rand(size(train_x',2)+1,N1)-1;
b2=2*rand(N1+1,N2)-1;
b=orth(2*rand(N2+1,N)'-1)';

C = 2^-30; s = .8;

fprintf(1,'N1= %d\n',N1);
fprintf(1,'N2= %d\n',N2);
fprintf(1,'N= %d\n',N);

[TrainingAccuracy,TestingAccuracy_RMSE,Training_time,Testing_time] = helm_regression_02(train_x, train_y, test_x, test_y, b1, b2, b, s, C);
RMSE_List(x,:) = TestingAccuracy_RMSE;
end

AccList = horzcat(RMSE_List);
headers = {'RMSE'};

csvwrite_with_headers(Result_File,AccList,headers)