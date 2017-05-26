%load 'input_data/ACTEWAGL/elm_input_wd.mat'; Result_File='optimal_data/ACTEWAGL/elm_optimal_wd.csv';
%load 'input_data/ACTEWAGL/elm_input_lwd.mat'; Result_File='optimal_data/ACTEWAGL/elm_optimal_lwd.csv';
%load 'input_data/ACTEWAGL/elm_input_lww.mat'; Result_File='optimal_data/ACTEWAGL/elm_optimal_lww.csv';
%load 'input_data/ACTEWAGL/elm_input_lwdw.mat'; Result_File='optimal_data/ACTEWAGL/elm_optimal_lwdw.csv';



No_of_Output=48;
NumberofHiddenNeurons=1;
ActivationFunction='sig';

RMSE_List = [];

for x = 1:300
    RMSE_temp = [];
    for xx = 1:20
        [TrainingTime, TestingTime, TrainingAccuracy_RMSE, TestingAccuracy_RMSE] = ELM_MultiOutputRegression(train_x, train_y, test_x, test_y, No_of_Output, NumberofHiddenNeurons, ActivationFunction);
        RMSE_temp(xx,:) = TestingAccuracy_RMSE;
    end
    
    RMSE_List(x,:) = mean(RMSE_temp);
    NumberofHiddenNeurons = NumberofHiddenNeurons + 1;
    
end

AccList = horzcat(RMSE_List);
headers = {'RMSE'};
csvwrite_with_headers(Result_File,AccList,headers)