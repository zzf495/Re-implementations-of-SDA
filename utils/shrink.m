function res = shrink(x,a)
   res=sign(x).*( max(abs(x)-a,0));
end