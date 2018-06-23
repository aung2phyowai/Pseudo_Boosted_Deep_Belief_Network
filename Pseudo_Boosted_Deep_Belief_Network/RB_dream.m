function [data_dream] = RB_dream(Data,W,h_Bias,v_Bias)
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here
     [dim,num]=size(Data);
      data_dream=Data;
      for i=1:num
  
          hid_result1=([Data(:,i);1]'*[W;h_Bias'])';
          hid_result1=sigmoid(hid_result1); % verticle;
          hid_dis_result1=discretize(hid_result1);

          imag_data=([hid_dis_result1',1]*[W';v_Bias'])';
          imag_data=sigmoid(imag_data);
          data_dream(:,i)=imag_data;

      end


end

