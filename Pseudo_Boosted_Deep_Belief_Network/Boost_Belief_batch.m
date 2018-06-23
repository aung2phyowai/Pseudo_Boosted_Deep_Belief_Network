[volu,dim,num]=size(batchdata);

disp('started');

L1=784;
L2=784;
L3=256;
cate=10;

time1=10;
time2=20;
time4=6;



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

num_clas=6;
data_weight=ones(volu,num,num_clas)/(volu*num);  % for the boosting exclusively; 
boost_factor=ones(1,num_clas); 

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







c_W1_cl=zeros((l1+1),l2,num_clas);
c_W2_cl=zeros((l2+1),l3,num_clas);
c_W3_cl=zeros((l3+1),l4,num_clas);
c_W4_cl=zeros((l4+1),10,num_clas);


Init_Epsi=Epsi;
pr_weight=data_weight(:,:,1);
orig_data=boost_data;
orig_targets=boost_targets;
new_batchdata=batchdata;
new_batchtargets=batchtargets;
for i=1:num_clas

    if i==1
    [c_W1_cl(:,:,i),c_W2_cl(:,:,i),c_W3_cl(:,:,i),c_W4_cl(:,:,i)] = Set_Classifier_batch(new_batchdata,new_batchtargets,c_W1,c_W2,c_W3,c_W4,time4);
    elseif i>1
    [c_W1_cl(:,:,i),c_W2_cl(:,:,i),c_W3_cl(:,:,i),c_W4_cl(:,:,i)] = Set_Classifier_batch(new_batchdata,new_batchtargets,c_W1_cl(:,:,(i-1)),c_W2_cl(:,:,(i-1)),c_W3_cl(:,:,(i-1)),c_W4_cl(:,:,(i-1)),time4);
    end
    
    [data_weight(:,:,i),boost_factor(i)]=Boost_Weight_batch(batchdata,batchtargets,pr_weight,c_W1_cl(:,:,i),c_W2_cl(:,:,i),c_W3_cl(:,:,i),c_W4_cl(:,:,i));
    
    disp('boost_factor(i)');
    disp(boost_factor(i));
    pr_weight=data_weight(:,:,i);
    
    prob_dist=data_weight(:,:,i)/sum(sum(data_weight(:,:,i)));
    
    prob_dist=prob_dist(:)';
    
    sample=discretesample(prob_dist,(volu*num));
    
    New_orig_data=orig_data(sample,:);
    New_orig_targets=orig_targets(sample,:);
    
    
    totnum=size(New_orig_data,1);
    rand('state',0); %so we know the permutation of the training data
    randomorder=randperm(totnum);
    new_batchdata=batchdata;
    new_batchtargets=batchtargets;
    
    for b=1:num
        new_batchdata(:,:,b) = New_orig_data(randomorder(1+(b-1)*volu:b*volu), :);
        new_batchtargets(:,:,b) = New_orig_targets(randomorder(1+(b-1)*volu:b*volu), :);
    end;
    
    
    orig_data=New_orig_data;
    orig_targets=New_orig_targets;
    
    disp('6');
    disp(i);

end



err_num=0;
for i=1:num

    this_num = committee_judge_batch(batchdata(:,:,i),batchtargets(:,:,i),c_W1_cl,c_W2_cl,c_W3_cl,c_W4_cl,cate,boost_factor);
    
    err_num=err_num+this_num;
    

end
    
error_rate=err_num/(volu*num);


[volu,dim,test_num]=size(testbatchdata);

test_err_num=0;
for i=1:test_num

    this_num = committee_judge_batch(testbatchdata(:,:,i),testbatchtargets(:,:,i),c_W1_cl,c_W2_cl,c_W3_cl,c_W4_cl,cate,boost_factor);
    
    test_err_num=test_err_num+this_num;
    

end
    


test_error_rate=test_err_num/(volu*test_num);









    