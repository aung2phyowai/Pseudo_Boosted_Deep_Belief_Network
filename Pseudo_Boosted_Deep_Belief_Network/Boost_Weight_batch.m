function [data_weight,boost_factor] = Boost_Weight_batch(batchdata,batchtargets,pr_weight,c_W1,c_W2,c_W3,c_W4)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here


    [volu,dim,num]=size(batchdata);
    correct=zeros(volu,num);
    
    err_weight=0;
    true_err_num=0;
    N=volu;
    counter=0;
    for batch = 1:num

        data = [batchdata(:,:,batch)];
        target = [batchtargets(:,:,batch)];
        data = [data ones(N,1)];
          
          
        w1probs = 1./(1 + exp(-data*c_W1)); w1probs = [w1probs  ones(N,1)];
        w2probs = 1./(1 + exp(-w1probs*c_W2)); w2probs = [w2probs ones(N,1)];
        w3probs = 1./(1 + exp(-w2probs*c_W3)); w3probs = [w3probs  ones(N,1)];
        targetout = exp(w3probs*c_W4);          % soft max;
        targetout = targetout./repmat(sum(targetout,2),1,10);
        

        [I J]=max(targetout,[],2);
        [I1 J1]=max(target,[],2);
         counter=counter+length(find(J==J1));
         true_err_num=true_err_num+length(find(J~=J1));
         correct(:,batch)=(J~=J1)';
         err_weight=err_weight+pr_weight(:,batch)'*correct(:,batch);
        

    end

    Err_Rate=err_weight/sum(sum(pr_weight));
    if Err_Rate==0
        Err_Rate=0.0001; 
        
    elseif Err_Rate==1
        Err_Rate=0.9;
        
    end
    
    
    true_error_ratio=true_err_num/(volu*num);
    disp('True_error_ratio');
    disp(true_error_ratio);
    
    
    
    boost_factor=log((1-Err_Rate)/Err_Rate);
    
    data_weight=pr_weight.*exp(correct);

    
end









