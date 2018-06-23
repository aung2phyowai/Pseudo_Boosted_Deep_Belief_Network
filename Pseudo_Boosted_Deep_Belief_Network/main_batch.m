
[volu,dim,num]=size(batchdata);

disp('started');

L1=784;
L2=784;
L3=256;
cate=10;

time1=10;
time2=20;
Iter_Size=60000;


Epsi=0.1;
mome=0.2;
we_cost=0.0002;

Init_W1=0.1*rand(dim,L1);
Init_W2=0.1*rand(L1,L2);
Init_W3=0.1*rand(L2,L3);

Init_h1_Bias=zeros(L1,1);
Init_v1_Bias=zeros(dim,1);
Init_h2_Bias=zeros(L2,1);
Init_v2_Bias=zeros(L1,1);
Init_h3_Bias=zeros(L3,1);
Init_v3_Bias=zeros(L2,1);
Init_h_B=zeros(cate,1);
Init_v_B=zeros(L3,1);


Init_Epsi=Epsi;
for i=1:time1

    [W1,h1_Bias,v1_Bias, Next_Data1] = RB_machine_batch(batchdata,Init_W1,Init_h1_Bias,Init_v1_Bias,Init_Epsi,mome,we_cost);
    
Init_W1=W1;
Init_h1_Bias=h1_Bias;
Init_v1_Bias=v1_Bias;
Init_Epsi=Epsi/sqrt(i*volu); % could be the possible reason of different performance;
disp('1');
disp(i);



end



Init_Epsi=Epsi;
for i=1:time1

[W2,h2_Bias,v2_Bias, Next_Data2] = RB_machine_batch(Next_Data1,Init_W2,Init_h2_Bias,Init_v2_Bias,Init_Epsi,mome,we_cost);


Init_W2=W2;
Init_h2_Bias=h2_Bias;
Init_v2_Bias=v2_Bias;
Init_Epsi=Epsi/sqrt(i*volu);

disp('2');
disp(i);


end


Init_Epsi=Epsi;
for i=1:time1

[W3,h3_Bias,v3_Bias, Next_Data3] = RB_machine_batch(Next_Data2,Init_W3,Init_h3_Bias,Init_v3_Bias,Init_Epsi,mome,we_cost);


Init_W3=W3;
Init_h3_Bias=h3_Bias;
Init_v3_Bias=v3_Bias;
Init_Epsi=Epsi/sqrt(i*volu);

disp('3');
disp(i);


end


c_W1=[W1;h1_Bias'];
c_W2=[W2;h2_Bias'];
c_W3=[W3;h3_Bias'];
c_W4=0.1*randn(size(c_W3,2)+1,10);


l1=size(c_W1,1)-1; 
l2=size(c_W2,1)-1;
l3=size(c_W3,1)-1;
l4=size(c_W4,1)-1;
l5=10; 






for epoch=1:time2
    
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
 
 if epoch<6
 
     N = size(data,1);
     XX = [data ones(N,1)];
     w1probs = 1./(1 + exp(-XX*c_W1)); w1probs = [w1probs  ones(N,1)];
     w2probs = 1./(1 + exp(-w1probs*c_W2)); w2probs = [w2probs ones(N,1)];
     w3probs = 1./(1 + exp(-w2probs*c_W3)); % w3probs = [w3probs  ones(N,1)];
     VV = [c_W4(:)']';
     Dim = [l4; l5];
     [X, fX] = minimize(VV,'CG_CLASSIFY_INIT',max_search,Dim,w3probs,targets);
     c_W4 = reshape(X,l4+1,l5);
     
     
 else
     
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
 


 end

 disp('4');
 disp(epoch);
 
end


error_rate = cal_correct_batch(batchdata,batchtargets,c_W1,c_W2,c_W3,c_W4);

test_error_rate = cal_correct_batch(testbatchdata,testbatchtargets,c_W1,c_W2,c_W3,c_W4);



