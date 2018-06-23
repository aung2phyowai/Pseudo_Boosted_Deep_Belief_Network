function [new_matrix] = deri_sigmoid(matrix)
%UNTITLED7 Summary of this function goes here
%   Detailed explanation goes here
    new_matrix=sigmoid(matrix).*(1-sigmoid(matrix));



end

