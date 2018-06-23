function [Err_Rate] = Cal_Correct(Data, Label, W1,B1,W2,B2,W3,B3)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
    [dim,num]=size(Data);
    
    err_num=0;
    for i=1:num

        This_Data=Data(:,i);

        Result1=([This_Data',1]*[W1;B1'])';
        Result1=sigmoid(Result1);
        
        Result2=([Result1',1]*[W2;B2'])';
        Result2=sigmoid(Result2);

        Result3=([Result2',1]*[W3;B3'])';
        Result3=soft_max(Result3);
        [value,index]=max(Result3);
        

        if index~=Label(i)+1
        
            err_num=err_num+1;
            
        end

    end
    
    Err_Rate=err_num/num;

end

