function m = mape(testY, pred)
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

err = abs(bsxfun(@minus, pred, testY));
pcterr = bsxfun(@rdivide, err, abs(testY));
m = mean(nanmean(pcterr,1));