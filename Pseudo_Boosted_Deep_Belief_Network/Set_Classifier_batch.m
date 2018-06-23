function [c_W1,c_W2,c_W3,c_W4] = Set_Classifier_batch(batchdata,batchtargets,c_W1,c_W2,c_W3,c_W4,Iter)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
[volu,dim,num]=size(batchdata);

l1=size(c_W1,1)-1; 
l2=size(c_W2,1)-1;
l3=size(c_W3,1)-1;
l4=size(c_W4,1)-1;
l5=10; 

for epoch=1:Iter
    
 countt=0;
 for batch = 1:num/10

 countt=countt+1;
 data=[];
 targets=[]; 
 for kk=1:10
  data=[data 
        batchdata(:,:,(countt-1)*10+kk)]; 
  targets=[targets
        batchtargets(:,:,(countt-1)*10+kk)];
 end

 max_search=3;
 

     
 VV = [c_W1(:)' c_W2(:)' c_W3(:)' c_W4(:)']';
 Dim = [l1; l2; l3; l4; l5];
     [X, fX] = minimize(VV,'CG_CLASSIFY',max_search,Dim,data,targets);
     
     c_W1 = reshape(X(1:(l1+1)*l2),l1+1,l2);
     xxx = (l1+1)*l2;
     c_W2 = reshape(X(xxx+1:xxx+(l2+1)*l3),l2+1,l3);
     xxx = xxx+(l2+1)*l3;
     c_W3 = reshape(X(xxx+1:xxx+(l3+1)*l4),l3+1,l4);
     xxx = xxx+(l3+1)*l4;
     c_W4 = reshape(X(xxx+1:xxx+(l4+1)*l5),l4+1,l5);


 end

 disp('6.1');
 disp(epoch);
 
end




end

