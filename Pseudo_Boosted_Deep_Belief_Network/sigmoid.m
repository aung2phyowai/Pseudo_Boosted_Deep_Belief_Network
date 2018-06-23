function [prob] = sigmoid(y)
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here

    prob=1./(1+exp(-y));

end

