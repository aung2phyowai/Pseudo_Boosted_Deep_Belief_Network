function [W,h_Bias,v_Bias, Next_Data] = RB_machine(Data,Init_W,Init_h_Bias,Init_v_Bias,Epsi,mome,we_cost)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
    [dim,num]=size(Data);
     next_dim=length(Init_h_Bias);

    W=Init_W; % vis*hid;
    h_Bias=Init_h_Bias; % verticle;
    v_Bias=Init_v_Bias; % verticle;
    hid_num=length(Init_h_Bias);
    vis_num=length(Init_v_Bias);
    W_inc=zeros(size(W));
    h_Bias_inc=zeros(size(h_Bias));
    v_Bias_inc=zeros(size(v_Bias));
    
    Epsi_Init=Epsi;
    for i=1:num

        Epsi=Epsi_Init; %/sqrt(i);
        This_Data=Data(:,i);
        hid_result1=([Data(:,i);1]'*[W;h_Bias'])';
        hid_result1=sigmoid(hid_result1); % verticle;
        hid_dis_result1=discretize(hid_result1); % verticle;
        

        imag_data=([hid_dis_result1',1]*[W';v_Bias'])';
        imag_data=sigmoid(imag_data);
        hid_result2=([imag_data;1]'*[W;h_Bias'])'; % verticle;
        hid_result2=sigmoid(hid_result2); % verticle;

        W_inc=mome*W_inc+Epsi*(This_Data*hid_result1'-imag_data*hid_result2'-we_cost*W);
        h_Bias_inc=mome*h_Bias_inc+Epsi*(hid_result1-hid_result2-we_cost*h_Bias);
        v_Bias_inc=mome*v_Bias_inc+Epsi*(This_Data-imag_data-we_cost*v_Bias);
        
        W=W+W_inc;
        h_Bias=h_Bias+h_Bias_inc;
        v_Bias=v_Bias+v_Bias_inc;
        

    end
    
    Next_Data=zeros(next_dim,num);
    
    for i=1:num
        Next_Data(:,i)=([Data(:,i);1]'*[W;h_Bias'])';
        Next_Data(:,i)=sigmoid(Next_Data(:,i));
    
    
    end


end























