function [ TT ] = relm_test( T, beta, ps )

HH = [T .1 * ones(size(T,1),1)];
clear T;
TT = HH * beta; TT  =  mapminmax('apply',TT',ps)';
clear HH;clear beta;

end

