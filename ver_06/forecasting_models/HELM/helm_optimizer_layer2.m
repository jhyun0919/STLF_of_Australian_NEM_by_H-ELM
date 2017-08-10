%load 'input_data/ACTEWAGL/helm_input_wd.mat'; Result_File='optimal_data/ACTEWAGL/helm_optimal_wd.csv';
%load 'input_data/ACTEWAGL/helm_input_lwd.mat'; Result_File='optimal_data/ACTEWAGL/helm_optimal_lwd.csv';
%load 'input_data/ACTEWAGL/helm_input_lww.mat'; Result_File='optimal_data/ACTEWAGL/helm_optimal_lww.csv';
%load 'input_data/ACTEWAGL/helm_input_lwdw.mat'; Result_File='optimal_data/ACTEWAGL/helm_optimal_lwdw.csv';

NumberofHiddenNeurons_layer_1=1;
NumberofHiddenNeurons_layer_2=1;
ActivationFunction='sig';

RMSE_List2 = []
for x = 1:300
    for xx = 1:300
        RMSE_temp = [];
        for xxx = 1:20
    	
        	N1=NumberofHiddenNeurons_layer_1;
    		N2=NumberofHiddenNeurons_layer_2;
            N=N2+1;
        	b1=2*rand(size(train_x',2)+1,N1)-1;
            b2=2*rand(N1+1,N2)-1;
    		b=orth(2*rand(N1+1,N)'-1)';
    		C = 2^-30; s = .8;

            [TrainingAccuracy_RMSE,TestingAccuracy_RMSE,Training_time,Testing_time] = helm_regression_02(train_x, train_y, test_x, test_y, b1, b2, b, s, C);
            RMSE_temp(xxx,:) = TestingAccuracy_RMSE;
        end
        RMSE_List1(xx,:) = mean(RMSE_temp); 
        NumberofHiddenNeurons_layer_2 = NumberofHiddenNeurons_layer_2 + 1;
    end
    RMSE_List2 = horzcat(RMSE_List2, RMSE_List1)
    NumberofHiddenNeurons_layer_1 = NumberofHiddenNeurons_layer_1 + 1;
    
end

AccList = horzcat(RMSE_List);
headers = {'RMSE'};
csvwrite_with_headers(Result_File,AccList,headers)