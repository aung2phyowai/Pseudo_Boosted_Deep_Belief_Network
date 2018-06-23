function [Imag] = Dis_Imag(Data,row,col)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

Imag=zeros(row,col);
[dim,col1]=size(Data);
%disp(dim);
%disp(row*col);
if dim~=row*col
    
    disp('dimension not equal');
else
    
    for i=1:row
        for j=1:col
        
            Imag(i,j)= Data((j-1)*row+i);
    
        end
    end
    
    
end




end

