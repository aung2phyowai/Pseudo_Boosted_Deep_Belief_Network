images = loadMNISTImages('train-images.idx3-ubyte');
labels = loadMNISTLabels('train-labels.idx1-ubyte');
images_test = loadMNISTImages('t10k-images.idx3-ubyte');
labels_test = loadMNISTLabels('t10k-labels.idx1-ubyte');

disp('started');

[row,column]=size(images);
[row_test,column_test]=size(images_test);


L1=256;
L2=100;
cate=10;

time1=2;
time2=2;    
Iter_Size=600;

Epsi=0.1;
mome=0.2;
we_cost=0.0002;

Data=images(:,1:Iter_Size);
Label=labels(1:Iter_Size);

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


Init_Epsi=Epsi;
for i=1:time1
    
[W3,B3] = Train_Layer(Next_Data2, Label, Init_W3, Init_B, Init_Epsi, mome,we_cost);

Init_W3=W3;
Init_B=B3;
Init_Epsi=Epsi/sqrt(i*Iter_Size);
disp('3');
disp(i);



end

Init_Epsi=Epsi;
for i=1:time2

[R_W1,R_B1,R_W2,R_B2,R_W3,R_B3] = Train_Network(Data,Label,W1,h1_Bias,W2,h2_Bias,W3,B3,Init_Epsi,mome,we_cost);


W1=R_W1;
h1_Bias=R_B1;
W2=R_W2;
h2_Bias=R_B2;
W3=R_W3;
B3=R_B3;


Init_Epsi=Epsi/sqrt(i*Iter_Size);

disp('4');
disp(i);


end

[Err_Rate] = Cal_Correct(Data, Label, R_W1,R_B1,R_W2,R_B2,R_W3,R_B3);

disp(Err_Rate);




