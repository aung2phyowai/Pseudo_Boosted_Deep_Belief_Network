function [ error_rate ] = cal_correct_batch(batchdata,batchtargets,c_W1,c_W2,c_W3,c_W4)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here


  [volu,dim,num]=size(batchdata);
  
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

      [I,J]=max(targetout,[],2);
      [I1,J1]=max(target,[],2);
      counter=counter+length(find(J==J1));
      
  end

  error_num=volu*num-counter;
  
  error_rate=error_num/(volu*num);
  
  

end

