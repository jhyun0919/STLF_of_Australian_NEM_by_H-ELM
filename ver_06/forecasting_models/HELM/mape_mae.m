function [MAPE, MAE] = mape_mae(testY, pred)
% Compute mean absolute percent error
%
% m = mape(actual, pred)
%
max_arg = 214794.796;
min_arg = 24477.404;

testY = testY';
pred = pred';

testY = testY * (max_arg - min_arg) + min_arg;
pred = pred * (max_arg - min_arg) + min_arg;


[row, col] = size(testY);

MAPE_List = [];
MAE_List = [];

for row_idx = 1:row
	mape_list = [];
	mae_list = [];
    for col_idx = 1:col
    	mae_temp = abs(testY(row_idx, col_idx) - pred(row_idx, col_idx));
    	mape_temp = 100 * mae_temp / abs(testY(row_idx, col_idx));
    	mape_list(col_idx, :) = mape_temp;
    	mae_list(col_idx, :) = mae_temp;
    end
    MAPE_List(row_idx, :) = mean(mape_list);
    MAE_List(row_idx, :) = mean(mae_list);
end

MAPE=mean(MAPE_List);
MAE=mean(MAE_List);