function [W1,B1,W2,B2,W3,B3] = Train_Network(Data,Label,Init_W1,Init_B1,Init_W2,Init_B2,Init_W3,Init_B3,Epsi,mome,we_cost)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
    [dim,num]=size(Data);
    [cate,col]=size(Init_B3);
    
    
    W1=Init_W1;
    B1=Init_B1;
    W2=Init_W2;
    B2=Init_B2;
    W3=Init_W3;
    B3=Init_B3;
    W1_inc=zeros(size(W1));
    B1_inc=zeros(size(B1));
    W2_inc=zeros(size(W2));
    B2_inc=zeros(size(B2));
    W3_inc=zeros(size(W3));
    B3_inc=zeros(size(B3));
    Epsi_Init=Epsi;
    
    for i=1:num

        Epsi=Epsi_Init/sqrt(i);
        This_Data=Data(:,i);
        This_Target=zeros(cate,1);
        This_Target(Label(i)+1)=1;
        Result1=([This_Data',1]*[Init_W1;Init_B1'])';
        Result1=sigmoid(Result1);
        
        Result2=([Result1',1]*[Init_W2;Init_B2'])';
        Result2=sigmoid(Result2);

        Result3=([Result2',1]*[Init_W3;Init_B3'])';
        Result3=soft_max(Result3);
        Diff3=This_Target-Result3;
        W3_inc=mome*W3_inc+Epsi*(Result2*Diff3'-we_cost*W3);
        B3_inc=mome*B3_inc+Epsi*(Diff3-we_cost*B3);
        
        Diff2=deri_sigmoid(Result2).*(Diff3'*W3')';
        W2_inc=mome*W2_inc+Epsi*(Result1*Diff2'-we_cost*W2);
        B2_inc=mome*B2_inc+Epsi*(Diff2-we_cost*B2);
        
        Diff1=deri_sigmoid(Result1).*(Diff2'*W2')';
        W1_inc=mome*W1_inc+Epsi*(This_Data*Diff1'-we_cost*W1);
        B1_inc=mome*B1_inc+Epsi*(Diff1-we_cost*B1);
        
        
        W3=W3+W3_inc;
        B3=B3+B3_inc;
        W2=W2+W2_inc;
        B2=B2+B2_inc;
        W1=W1+W1_inc;
        B1=B1+B1_inc;
        
        



    end

end











