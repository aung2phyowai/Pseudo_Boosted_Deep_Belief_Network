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

time1=10;
time2=5;
time4=5;
Iter_Size=10000;
number_cl=6;

Epsi=0.1;
mome=0.2;
we_cost=0.0002;

Data=images(:,1:Iter_Size);
Label=labels(1:Iter_Size);

data_weight=ones(Iter_Size,number_cl)/Iter_Size;  % for the boosting exclusively; 
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


B1=h1_Bias;
B2=h2_Bias;
W1_cl=zeros(row,L1,number_cl);
B1_cl=zeros(L1,number_cl);
W2_cl=zeros(L1,L2,number_cl);
B2_cl=zeros(L2,number_cl);
W3_cl=zeros(L2,cate,number_cl);
B3_cl=zeros(cate,number_cl);

Init_Epsi=Epsi;
pr_weight=data_weight(:,1);
New_Data=Data;
New_Label=Label;
for i=1:number_cl
if i==1
    [W1_cl(:,:,i),B1_cl(:,i),W2_cl(:,:,i),B2_cl(:,i),W3_cl(:,:,i),B3_cl(:,i)] = Set_Classifier(New_Data,New_Label,W1,B1,W2,B2,L2,cate,Init_Epsi,mome,we_cost,time4);
elseif i>1
    [W1_cl(:,:,i),B1_cl(:,i),W2_cl(:,:,i),B2_cl(:,i),W3_cl(:,:,i),B3_cl(:,i)] = Set_Classifier(New_Data,New_Label,W1_cl(:,:,(i-1)),B1_cl(:,(i-1)),W2_cl(:,:,(i-1)),B2_cl(:,(i-1)),L2,cate,Init_Epsi,mome,we_cost,time4);
end
disp('3');
disp(i);
[data_weight(:,i),boost_factor(i)]=Boost_Weight(Data,Label,pr_weight,W1_cl(:,:,i),B1_cl(:,i),W2_cl(:,:,i),B2_cl(:,i),W3_cl(:,:,i),B3_cl(:,i));
disp(boost_factor(i));
pr_weight=data_weight(:,i);


prob_dist=data_weight(:,i)/sum(data_weight(:,i));

sample=discretesample(prob_dist,Iter_Size);

for j=1:Iter_Size

    New_Data(:,j)=Data(:,sample(j));
    New_Label(j)=Label(sample(j));

end




end

err_num=0;
for i=1:Iter_Size

    [Index] = committee_judge(Data(:,i),W1_cl,B1_cl,W2_cl,B2_cl,W3_cl,B3_cl,cate,boost_factor);
    
    if Index~=Label(i)+1
        
        err_num=err_num+1;
        
    end


end

err_rate=err_num/Iter_Size;
disp('err_rate')
disp(err_rate);








