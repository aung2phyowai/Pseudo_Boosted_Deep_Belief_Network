function [W,B] = Train_Layer(Data, Target, Init_W, Init_B, Epsi, mome,we_cost)
%UNTITLED5 Summary of this function goes here
%   Detailed explanation goes here
    [dim1,num]=size(Data);
    [cate,col]=size(Init_B);
    
    W=Init_W;
    B=Init_B;
    W_inc=zeros(size(W));
    B_inc=zeros(size(B));
    
    Epsi_Init=Epsi;
    for i=1:num
        
        Epsi=Epsi_Init/sqrt(i);
        This_Data=Data(:,i);
        True_Target=zeros(cate,1);
        True_Target(Target(i)+1)=1;

        Esti_Target=([Data(:,i)',1]*[W;B'])';
        Esti_Target=soft_max(Esti_Target);
        Diff=True_Target-Esti_Target;
        W_inc=mome*W_inc+Epsi*(This_Data*Diff'-we_cost*W);
        B_inc=mome*B_inc+Epsi*(Diff-we_cost*B);
        
        W=W+W_inc;
        B=B+B_inc;
        
    end


end





