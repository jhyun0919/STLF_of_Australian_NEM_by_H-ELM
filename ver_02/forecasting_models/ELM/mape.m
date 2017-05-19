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

[row, col] = size(testY);
AccList = [];
for x = 1:row
    err = abs(bsxfun(@minus, pred, testY));
    pcterr = bsxfun(@rdivide, err, abs(testY));
    m_ = mean(nanmean(pcterr,1));
    AccList(x,:) = m_;
end

m=mean(m_);