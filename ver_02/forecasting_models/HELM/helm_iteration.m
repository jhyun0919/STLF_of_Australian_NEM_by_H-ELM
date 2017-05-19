%load helm_input_data00.mat; Result_File='helm_forecast_result00.csv'; NumberofHiddenNeurons=25;
%load helm_input_data01.mat; Result_File='helm_forecast_result01.csv'; NumberofHiddenNeurons=40;
%load helm_input_data02.mat; Result_File='helm_forecast_result02.csv'; NumberofHiddenNeurons=30;
load helm_input_data03.mat; Result_File='helm_forecast_result03.csv'; NumberofHiddenNeurons=30;

N1=NumberofHiddenNeurons;
N2=NumberofHiddenNeurons;
N3=NumberofHiddenNeurons;
N4=NumberofHiddenNeurons;
N=N4+1;

AccList = [];

for x = 1:1000
x
b1=2*rand(size(train_x',2)+1,N1)-1;
b2=2*rand(N1+1,N2)-1;
b3=2*rand(N2+1,N3)-1;
b4=2*rand(N3+1,N4)-1;
b=orth(2*rand(N4+1,N)'-1)';

C = 2^-30; s = .8;

fprintf(1,'N1= %d\n',N1);
fprintf(1,'N2= %d\n',N2);
fprintf(1,'N3= %d\n',N3);
fprintf(1,'N4= %d\n',N4);
fprintf(1,'N= %d\n',N);

[TrainingAccuracy,TestingAccuracy,Training_time,Testing_time] = helm_regression(train_x, train_y, test_x, test_y, b1, b2, b3, b4, b, s, C);
AccList(x,:) = TestingAccuracy;

end

headers = {'MAPE'};

csvwrite_with_headers(Result_File,AccList,headers)