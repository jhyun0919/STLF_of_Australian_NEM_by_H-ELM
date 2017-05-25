function [ T, beta, ps ] = relm_train( X, b )
	
H = [X .1 * ones(size(X,1),1)];
clear X;
A = H * b;A = mapminmax(A);
clear b;
beta  =  sparse_elm_autoencoder(A,H,1e-3,50)';
clear A;
T = H * beta;
[T,ps]  =  mapminmax(T',0,1);T = T';
clear H;
end
