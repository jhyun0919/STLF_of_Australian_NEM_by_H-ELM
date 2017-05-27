function [Training_time,Testing_time, TrainingAccuracy_RMSE, TestingAccuracy_RMSE, TrainingAccuracy_MAPE, TestingAccuracy_MAPE, TrainingAccuracy_MAE, TestingAccuracy_MAE] = helm_regression_(train_x,train_y,test_x,test_y,b1,b2,b,s,C)

%%%%%%%%%%%%%%%%%%%%%%%
% training part
%%%%%%%%%%%%%%%%%%%%%%%

tic
train_x = zscore(train_x)';

%% 1st layer RELM
[T1, beta1, ps1] = relm_train(train_x, b1);
clear train_x; clear b1;
fprintf(1,'Layer 1: Max Val of Output %f Min Val %f\n',max(T1(:)),min(T1(:)));

%% 2nd layer RELM
[T2, beta2, ps2] = relm_train(T1, b2);
clear T1; clear b2;
fprintf(1,'Layer 2: Max Val of Output %f Min Val %f\n',max(T2(:)),min(T2(:)));

%% 3rd layer RELM
%[T3, beta3, ps3] = relm_train(T2, b3);
%clear T2; clear b3;
%fprintf(1,'Layer 3: Max Val of Output %f Min Val %f\n',max(T3(:)),min(T3(:)));

%% 4rd layer RELM
%[T4, beta4, ps4] = relm_train(T3, b4);
%clear T3; clear b4;
%fprintf(1,'Layer 4: Max Val of Output %f Min Val %f\n',max(T4(:)),min(T4(:)));

%% Original ELM regressor
[T, l, beta] = elm_train(train_y, T2, b, s, C);
clear T2;
fprintf(1,'ELM Layer: Max Val of Output %f Min Val %f\n',l,min(T(:)));

%% Finsh Training
Training_time = toc;
%% Calculate the training accuracy
predict_y = T * beta;
clear T;

disp('Training has been finished!');
disp(['The Total Training Time is : ', num2str(Training_time), ' seconds' ]);

TrainingAccuracy_RMSE=sqrt(mse(train_y' - predict_y));               %   Calculate training accuracy (RMSE) for regression case
[TrainingAccuracy_MAPE, TrainingAccuracy_MAE]=mape_mae(train_y',predict_y);
clear H;

disp(['Training Accuracy RMSE= ', num2str(TrainingAccuracy_RMSE)]);
disp(['Training Accuracy MAPE= ', num2str(TrainingAccuracy_MAPE)]);
disp(['Training Accuracy MAE= ', num2str(TrainingAccuracy_MAE)]);



%%%%%%%%%%%%%%%%%%%%%%%
% testing part
%%%%%%%%%%%%%%%%%%%%%%%

tic;
test_x = zscore(test_x)';

%% 1st layer feedforward
TT1 = relm_test(test_x, beta1, ps1);
clear test_x;

%% 2nd layer feedforward
TT2 = relm_test(TT1, beta2, ps2);
clear TT1;

%% 3rd layer feedforward
%TT3 = relm_test(TT2, beta3, ps3);
%clear TT2;

%% 4th layer feedforward
%TT4 = relm_test(TT3, beta4, ps4);
%clear TT3;

%% Last layer feedforward
predict_yy = elm_test(TT2, beta, l, b);
clear TT2; clear b;

%% Calculate the testing accuracy
Testing_time = toc;
TestingAccuracy_RMSE=sqrt(mse(test_y' - predict_yy));            %   Calculate testing accuracy (RMSE) for regression case
[TestingAccuracy_MAPE, TestingAccuracy_MAE]=mape_mae(test_y',predict_yy);

disp('Testing has been finished!');
disp(['The Total Testing Time is : ', num2str(Testing_time), ' seconds' ]);

disp(['Testing Accuracy RMSE= ', num2str(TestingAccuracy_RMSE)]);
disp(['Testing Accuracy MAPE= ', num2str(TestingAccuracy_MAPE)]);
disp(['Testing Accuracy MAE= ', num2str(TestingAccuracy_MAE)]);

fprintf('\n');
