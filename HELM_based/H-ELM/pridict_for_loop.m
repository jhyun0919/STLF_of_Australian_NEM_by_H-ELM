%load helm_input_data_preprocessed.mat;
load helm_input_data_raw.mat;

N1=50;
N2=50;
N3=50;
N4=50;
N=N4+1;

b1=2*rand(size(train_x',2)+1,N1)-1;
b2=2*rand(N1+1,N2)-1;
b3=2*rand(N2+1,N3)-1;
b4=2*rand(N3+1,N4)-1;
b=orth(2*rand(N4+1,N)'-1)';

C = 2^-30; s = .8;

fprintf(1,'N1= %d\n',N1);
fprintf(1,'N2= %d\n',N2);
fprintf(1,'N3= %d\n',N3);
fprintf(1,'N4= %d\n',N4);
fprintf(1,'N= %d\n',N);

AccList = [];

for n = 1:5
    rand('state', n);
    
    Acc = helm_regression(train_x, train_y, test_x, test_y, b1, b2, b3, b4, b, s, C);
    AccList = [AccList , Acc];
end