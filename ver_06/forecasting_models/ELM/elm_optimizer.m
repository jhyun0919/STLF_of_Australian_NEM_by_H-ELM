%load 'input_data/ACTEWAGL/elm_input_wd.mat'; Result_File='optimal_data/ACTEWAGL/elm_optimal_wd.csv';
%load 'input_data/ACTEWAGL/elm_input_lwd.mat'; Result_File='optimal_data/ACTEWAGL/elm_optimal_lwd.csv';
%load 'input_data/ACTEWAGL/elm_input_lww.mat'; Result_File='optimal_data/ACTEWAGL/elm_optimal_lww.csv';
load 'input_data/ACTEWAGL/elm_input_lwdw.mat'; Result_File='optimal_data/ACTEWAGL/elm_optimal_lwdw.csv';



No_of_Output=48;
NumberofHiddenNeurons=1;
ActivationFunction='sig';

RMSE_List = [];
MAPE_List = [];
MAE_List = [];

for x = 1:300
    RMSE_temp = [];
    MAPE_temp = [];
    MAE_temp = [];
    for xx = 1:20
        [TrainingTime, TestingTime, TrainingAccuracy_RMSE, TestingAccuracy_RMSE, TrainingAccuracy_MAPE, TestingAccuracy_MAPE, TrainingAccuracy_MAE, TestingAccuracy_MAE] = ELM_MultiOutputRegression(train_x, train_y, test_x, test_y, No_of_Output, NumberofHiddenNeurons, ActivationFunction);
        RMSE_temp(xx,:) = TestingAccuracy_RMSE;
        MAPE_temp(xx,:) = TestingAccuracy_MAPE;
        MAE_temp(xx,:) = TestingAccuracy_MAE;
    end
    
    RMSE_List(x,:) = mean(RMSE_temp);
    MAPE_List(x,:) = mean(MAPE_temp);
    MAE_List(x,:) = mean(MAE_temp);
    NumberofHiddenNeurons = NumberofHiddenNeurons + 1;
    
end

AccList = horzcat(RMSE_List, MAPE_List,MAE_List);
headers = {'RMSE', 'MAPE', 'MAE'};
csvwrite_with_headers(Result_File,AccList,headers)