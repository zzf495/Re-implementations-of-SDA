function res = shrink(x,a)
%% input:
%%%		X: the matrix, m*n
%%%		a: the shinrkage paramter
%% output:
%%%		res:	the results, m*n
   res=sign(x).*( max(abs(x)-a,0));
end