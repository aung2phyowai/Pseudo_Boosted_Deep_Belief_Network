function [Index] = committee_judge(Data,W1_cl,B1_cl,W2_cl,B2_cl,W3_cl,B3_cl,cate,boost_factor)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

cl_number=length(boost_factor);
This_Data=Data;
Accu_Conf=zeros(cate,1);

for i=1:cl_number

    Result1=([This_Data',1]*[W1_cl(:,:,i);B1_cl(:,i)'])';
    Result1=sigmoid(Result1);
        
    Result2=([Result1',1]*[W2_cl(:,:,i);B2_cl(:,i)'])';
    Result2=sigmoid(Result2);

    Result3=([Result2',1]*[W3_cl(:,:,i);B3_cl(:,i)'])';
    Result3=soft_max(Result3);
    Accu_Conf=Accu_Conf+boost_factor(i)*Result3;
    

end


[Value,Index]=max(Accu_Conf);






end

