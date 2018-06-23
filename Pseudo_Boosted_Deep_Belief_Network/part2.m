

B1=h1_Bias;
B2=h2_Bias;
W1_cl=zeros(row,L1,number_cl);
B1_cl=zeros(L1,number_cl);
W2_cl=zeros(L1,L2,number_cl);
B2_cl=zeros(L2,number_cl);
W3_cl=zeros(L2,cate,number_cl);
B3_cl=zeros(cate,number_cl);

Iter_Size=10000;
Data=images(:,1:Iter_Size);
Label=labels(1:Iter_Size);

number_cl=4;
time4=10;
data_weight=ones(Iter_Size,number_cl)/Iter_Size;
boost_factor=ones(1,number_cl);


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

