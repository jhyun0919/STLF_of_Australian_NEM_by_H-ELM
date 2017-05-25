function MAPE = mape(testY, pred)
% Compute mean absolute percent error
%
% m = mape(actual, pred)
%
% actual is a column vector of actual values
% pred is a matrix of predictions (one per column)
%
% m is the mean absolute percent error (ignoring NaNs) for each column of
% pred. 

% Copyright 2014-2015 The MathWorks, Inc.

testY = testY';
pred = pred';

[row, col] = size(testY);
MAPE_List = [];
for row_idx = 1:row
	err_list = [];
    for col_idx = 1:col
    	err_list(col_idx, :) = abs(testY(row_idx, col_idx) - pred(row_idx, col_idx)) / abs(testY(row_idx, col_idx));
    end
    MAPE_List(row_idx, :) = mean(err_list);
end

MAPE=mean(MAPE_List);