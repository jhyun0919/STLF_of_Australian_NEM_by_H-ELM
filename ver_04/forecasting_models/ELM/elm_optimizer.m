%load 'input_data/elm_input_d.mat'; Result_File='optimal_data/elm_optimal_d.csv';
%load 'input_data/elm_input_w.mat'; Result_File='optimal_data/elm_optimal_w.csv';
%load 'input_data/elm_input_dw.mat'; Result_File='optimal_data/elm_optimal_dw.csv';
load 'input_data/elm_input.mat'; Result_File='optimal_data/elm_optimal.csv';

No_of_Output=48*7;
NumberofHiddenNeurons=1;
ActivationFunction='sig';

RMSE_List = [];

for x = 1:100
    RMSE_temp = [];
    MAPE_temp = [];
    for xx = 1:30
        [TrainingTime, TestingTime, TrainingAccuracy_RMSE, TestingAccuracy_RMSE] = ELM_MultiOutputRegression(train_x, train_y, test_x, test_y, No_of_Output, NumberofHiddenNeurons, ActivationFunction);
        RMSE_temp(xx,:) = TestingAccuracy_RMSE;
    end
    
    RMSE_List(x,:) = mean(RMSE_temp);
    NumberofHiddenNeurons = NumberofHiddenNeurons + 1;
    
end

AccList = horzcat(RMSE_List);
headers = {'RMSE'};
csvwrite_with_headers(Result_File,AccList,headers)