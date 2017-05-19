%load elm_input_data00.mat; Result_File='elm_optimal_result00.csv';
%load elm_input_data01.mat; Result_File='elm_optimal_result01.csv';
%load elm_input_data02.mat; Result_File='elm_optimal_result02.csv';
load elm_input_data03.mat; Result_File='elm_optimal_result03.csv';

No_of_Output=48;
NumberofHiddenNeurons=1;
ActivationFunction='sig';

AccList = [];

for x = 1:200
    Acctemp = [];
    for xx = 1:20
        [TrainingTime, TestingTime, TrainingAccuracy, TestingAccuracy] = ELM_MultiOutputRegression(train_x, train_y, test_x, test_y, No_of_Output, NumberofHiddenNeurons, ActivationFunction);
        Acctemp(xx,:) = TestingAccuracy;
    end
    
    AccList(x,:) = mean(Acctemp);
    NumberofHiddenNeurons = NumberofHiddenNeurons + 1;
    
end

headers = {'MAPE'};

csvwrite_with_headers(Result_File,AccList,headers)