function y = L2Norm(x)
% x is a feature matrix: one example in a row
% add 1e-4
y = x./repmat(1e-4+sqrt(sum(x.^2,2)),[1 size(x,2)]);