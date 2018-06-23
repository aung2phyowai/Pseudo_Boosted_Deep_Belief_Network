function [data_weight,boost_factor] = Boost_Weight(Data,Label,pr_weight,W1,B1,W2,B2,W3,B3)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here


    [dim,num]=size(Data);
    correct=zeros(1,num);
    data_weight=pr_weight;
    
    err_weight=0;
    true_err_num=0;
    for i=1:num

        This_Data=Data(:,i);
        %disp(size(W1));
        %disp(size(B1));
        Result1=([This_Data',1]*[W1;B1'])';
        Result1=sigmoid(Result1);
        
        Result2=([Result1',1]*[W2;B2'])';
        Result2=sigmoid(Result2);

        Result3=([Result2',1]*[W3;B3'])';
        Result3=soft_max(Result3);
        [value,index]=max(Result3);
        

        if index~=(Label(i)+1)
        
            correct(i)=1;
            err_weight=err_weight+pr_weight(i);
            true_err_num=true_err_num+1;
            
        end

    end

    Err_Rate=err_weight/sum(pr_weight);
    if Err_Rate==0
        Err_Rate=0.0001; 
        
    elseif Err_Rate==1
        Err_Rate=0.9;
        
    end
    
    
    true_error_ratio=true_err_num/num;
    disp('True_error_ratio');
    disp(true_error_ratio);
    
    
        
    boost_factor=log((1-Err_Rate)/Err_Rate);
    

    for i=1:num

        data_weight(i)=pr_weight(i)*exp(correct(i));


    end

end

