function [ T, l, beta ] = elm__train( train_y, X, b, s, C)

H = [X .1 * ones(size(X,1),1)];
clear X;

T = H * b;
l = max(max(T));l = s/l;

T = tansig(T * l);
clear H;

beta = (T'  *  T+eye(size(T',1)) * (C)) \ ( T'  *  train_y');

end

