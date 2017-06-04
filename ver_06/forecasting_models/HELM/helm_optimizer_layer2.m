%load 'input_data/ACTEWAGL/helm_input_wd.mat'; Result_File='optimal_data/ACTEWAGL/layer2/helm_optimal_wd.csv';
%load 'input_data/ACTEWAGL/helm_input_lwd.mat'; Result_File='optimal_data/ACTEWAGL/layer2/helm_optimal_lwd.csv';
%load 'input_data/ACTEWAGL/helm_input_lww.mat'; Result_File='optimal_data/ACTEWAGL/layer2/helm_optimal_lww.csv';
%load 'input_data/ACTEWAGL/helm_input_lwdw1.mat'; Result_File='optimal_data/ACTEWAGL/layer2/helm_optimal_lwdw1.csv';
%load 'input_data/ACTEWAGL/helm_input_lwdw2.mat'; Result_File='optimal_data/ACTEWAGL/layer2/helm_optimal_lwdw2.csv';
%load 'input_data/ACTEWAGL/helm_input_lwdw3.mat'; Result_File='optimal_data/ACTEWAGL/layer2/helm_optimal_lwdw3.csv';
load 'input_data/ACTEWAGL/helm_input_lwdw4.mat'; Result_File='optimal_data/ACTEWAGL/layer2/helm_optimal_lwdw4.csv';
%load 'input_data/ACTEWAGL/helm_input_lwdw5.mat'; Result_File='optimal_data/ACTEWAGL/layer2/helm_optimal_lwdw5.csv';
%load 'input_data/ACTEWAGL/helm_input_lwdw6.mat'; Result_File='optimal_data/ACTEWAGL/layer2/helm_optimal_lwdw6.csv';



NumberofHiddenNeurons_layer_1=1;
NumberofHiddenNeurons_layer_2=1;
ActivationFunction='sig';


RMSE_List = [];
MAPE_List = [];
MAE_List = [];

for x = 1:300
    RMSE_temp = [];
    MAPE_temp = [];
    MAE_temp = [];
    for xx = 1:10
        x
        N1=NumberofHiddenNeurons_layer_1;
        N2=NumberofHiddenNeurons_layer_2;
        N=N2+1;
        b1=2*rand(size(train_x',2)+1,N1)-1;
        b2=2*rand(N1+1,N2)-1;
        b=orth(2*rand(N1+1,N)'-1)';
        C = 2^-30; s = .8;

        [Training_time,Testing_time, TrainingAccuracy_RMSE, TestingAccuracy_RMSE, TrainingAccuracy_MAPE, TestingAccuracy_MAPE, TrainingAccuracy_MAE, TestingAccuracy_MAE] = helm_regression_02(train_x, train_y, test_x, test_y, b1, b2, b, s, C);
        RMSE_temp(xx,:) = TestingAccuracy_RMSE;
        MAPE_temp(xx,:) = TestingAccuracy_MAPE;
        MAE_temp(xx,:) = TestingAccuracy_MAE;
    end
    
    RMSE_List(x,:) = mean(RMSE_temp);
    MAPE_List(x,:) = mean(MAPE_temp);
    MAE_List(x,:) = mean(MAE_temp);
    NumberofHiddenNeurons_layer_1 = NumberofHiddenNeurons_layer_1 + 1;
    NumberofHiddenNeurons_layer_2 = NumberofHiddenNeurons_layer_2 + 1;
    
end

AccList = horzcat(RMSE_List, MAPE_List,MAE_List);
headers = {'RMSE', 'MAPE', 'MAE'};
csvwrite_with_headers(Result_File,AccList,headers)