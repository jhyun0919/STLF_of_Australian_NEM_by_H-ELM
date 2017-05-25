function [ predict_yy ] = elm_test( T, beta, l, b )

HH = [T .1 * ones(size(T,1),1)];
clear T;
TT = tansig(HH * b * l);
clear HH;clear b;

predict_yy = TT * beta;

end

