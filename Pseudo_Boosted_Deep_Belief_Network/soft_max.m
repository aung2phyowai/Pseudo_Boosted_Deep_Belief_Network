function [Y] = soft_max(X)
%UNTITLED12 Summary of this function goes here
%   Detailed explanation goes here

deno=sum(exp(X));
Y=exp(X)/deno;

end



