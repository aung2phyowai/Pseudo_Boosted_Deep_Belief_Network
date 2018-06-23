function [W,h_Bias,v_Bias, Next_Data] = RB_machine_batch(Data,Init_W,Init_h_Bias,Init_v_Bias,Epsi,mome,we_cost)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
    [volu,dim,num]=size(Data);
     next_dim=length(Init_h_Bias);

    W=Init_W; % vis*hid;
    h_Bias=Init_h_Bias; % verticle;
    v_Bias=Init_v_Bias; % verticle;
    hid_num=length(Init_h_Bias);
    vis_num=length(Init_v_Bias);
    W_inc=zeros(size(W));
    h_Bias_inc=zeros(size(h_Bias));
    v_Bias_inc=zeros(size(v_Bias));
    Bias=ones(1,volu);
    
    Epsi_Init=Epsi;
    for i=1:num

        Epsi=Epsi_Init; %/sqrt(i);
        This_Data=Data(:,:,i)';
        hid_result1=([This_Data;Bias]'*[W;h_Bias'])';
        hid_result1=sigmoid(hid_result1); % verticle;
        hid_dis_result1=discretize(hid_result1); % verticle;
        

        imag_data=([hid_dis_result1',Bias']*[W';v_Bias'])';
        imag_data=sigmoid(imag_data);
        hid_result2=([imag_data;Bias]'*[W;h_Bias'])'; % verticle;
        hid_result2=sigmoid(hid_result2); % verticle;
        
        
        Diff=This_Data*hid_result1'-imag_data*hid_result2';
        hid_diff=sum(hid_result1')'-sum(hid_result2')';
        vis_diff=sum(This_Data')'-sum(imag_data')';
        W_inc=mome*W_inc+Epsi*(Diff/volu-we_cost*W);
        h_Bias_inc=mome*h_Bias_inc+Epsi*(hid_diff/volu); %-we_cost*h_Bias
        v_Bias_inc=mome*v_Bias_inc+Epsi*(vis_diff/volu); %-we_cost*v_Bias
        
        W=W+W_inc;
        h_Bias=h_Bias+h_Bias_inc;
        v_Bias=v_Bias+v_Bias_inc;
        

    end
    
    Next_Data=zeros(volu,next_dim,num);
    
    for i=1:num
        Next_Data(:,:,i)=[Data(:,:,i)';Bias]'*[W;h_Bias'];
        Next_Data(:,:,i)=sigmoid(Next_Data(:,:,i));
    
    
    end


end























