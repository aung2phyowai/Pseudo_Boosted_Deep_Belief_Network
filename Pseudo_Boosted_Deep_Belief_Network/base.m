images = loadMNISTImages('train-images.idx3-ubyte');
labels = loadMNISTLabels('train-labels.idx1-ubyte');
images_test = loadMNISTImages('t10k-images.idx3-ubyte');
labels_test = loadMNISTLabels('t10k-labels.idx1-ubyte');

clear;
clc;

disp('started');

[row,column]=size(images);
[row_test,column_test]=size(images_test);


L1=256;
L2=100;
cate=10;

time1=20;
time2=5;    
time4=5;
Iter_Size=60000;
number_cl=6;

Epsi=0.1;
mome=0.2;
we_cost=0.0002;

Data=images(:,1:Iter_Size);
Label=labels(1:Iter_Size);

data_weight=ones(Iter_Size,number_cl)/Iter_Size;
boost_factor=ones(1,number_cl);


Init_W1=0.1*rand(row,L1);
Init_W2=0.1*rand(L1,L2);
Init_W3=0.1*rand(L2,cate);
Init_h1_Bias=zeros(L1,1);
Init_v1_Bias=zeros(row,1);
Init_h2_Bias=zeros(L2,1);
Init_v2_Bias=zeros(L1,1);
Init_B=zeros(cate,1);



Init_Epsi=Epsi;
for i=1:time1

[W1,h1_Bias,v1_Bias, Next_Data1] = RB_machine(Data,Init_W1,Init_h1_Bias,Init_v1_Bias,Init_Epsi,mome,we_cost);

Init_W1=W1;
Init_h1_Bias=h1_Bias;
Init_v1_Bias=v1_Bias;
Init_Epsi=Epsi/sqrt(i*Iter_Size);
disp('1');
disp(i);


end

Init_Epsi=Epsi;
for i=1:time1

[W2,h2_Bias,v2_Bias, Next_Data2] = RB_machine(Next_Data1,Init_W2,Init_h2_Bias,Init_v2_Bias,Init_Epsi,mome,we_cost);


Init_W2=W2;
Init_h2_Bias=h2_Bias;
Init_v2_Bias=v2_Bias;
Init_Epsi=Epsi/sqrt(i*Iter_Size);

disp('2');
disp(i);


end
