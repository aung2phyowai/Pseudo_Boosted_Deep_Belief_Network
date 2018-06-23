function this_num = committee_judge_batch(this_batch,this_targets,c_W1_cl,c_W2_cl,c_W3_cl,c_W4_cl,cate,boost_factor)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
[volu,dim]=size(this_batch);
cl_number=length(boost_factor);
data=this_batch;
target=this_targets;
N=volu;
Accu_Conf=zeros(volu,cate);
data = [data ones(N,1)];

for i=1:cl_number
    
    %disp(size(data));
    %disp(size(c_W1_cl(:,:,i)));
    w1probs = 1./(1 + exp(-data*c_W1_cl(:,:,i))); w1probs = [w1probs  ones(N,1)];
    w2probs = 1./(1 + exp(-w1probs*c_W2_cl(:,:,i))); w2probs = [w2probs ones(N,1)];
    w3probs = 1./(1 + exp(-w2probs*c_W3_cl(:,:,i))); w3probs = [w3probs  ones(N,1)];
    targetout = exp(w3probs*c_W4_cl(:,:,i));          % soft max;
    targetout = targetout./repmat(sum(targetout,2),1,10);
    Accu_Conf=Accu_Conf+boost_factor(i)*targetout;
    

end


[I,J]=max(Accu_Conf,[],2);
[I1,J1]=max(target,[],2);

this_num=length(find(J~=J1));




end

