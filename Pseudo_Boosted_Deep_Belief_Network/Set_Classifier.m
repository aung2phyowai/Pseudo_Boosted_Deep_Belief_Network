function [W1,B1,W2,B2,W3,B3] = Set_Classifier(Data,Label,W1,B1,W2,B2,L2,cate,Init_Epsi,mome,we_cost,Iter)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
Init_W3=0.1*rand(L2,cate);
Init_B=zeros(cate,1);
[dim,num]=size(Data);
Data_Next=zeros(L2,num);

for i=1:num
    This_Data=Data(:,i);
    Result1=([This_Data',1]*[W1;B1'])';
    Result1=sigmoid(Result1);
    Result2=([Result1',1]*[W2;B2'])';
    Data_Next(:,i)=sigmoid(Result2);
    
    
end

Epsi=Init_Epsi;
Iter_Size=num;
for i=1:Iter

[W3,B3] = Train_Layer(Data_Next, Label, Init_W3, Init_B, Init_Epsi, mome,we_cost);
Init_W3=W3;
Init_B=B3;
Init_Epsi=Epsi/sqrt(i*Iter_Size);


end


Init_Epsi=Epsi;
for i=1:Iter

[R_W1,R_B1,R_W2,R_B2,R_W3,R_B3] = Train_Network(Data,Label,W1,B1,W2,B2,W3,B3,Init_Epsi,mome,we_cost);

W1=R_W1;
B1=R_B1;
W2=R_W2;
B2=R_B2;
W3=R_W3;
B3=R_B3;

Init_Epsi=Epsi/sqrt(i*Iter_Size);




end




end

