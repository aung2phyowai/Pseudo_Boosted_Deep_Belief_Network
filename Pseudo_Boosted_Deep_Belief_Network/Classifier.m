function [Result] = Classifier(Data,W1,B1,W2,B2,W3,B3)
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here
This_Data=Data;
Result1=([This_Data',1]*[W1;B1'])';
Result1=sigmoid(Result1);
        
Result2=([Result1',1]*[W2;B2'])';
Result2=sigmoid(Result2);

Result3=([Result2',1]*[W3;B3'])';
Result3=soft_max(Result3);
[value,index]=max(Result3);
Result=index;



end

