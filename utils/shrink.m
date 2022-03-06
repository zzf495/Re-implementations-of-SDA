function res = shrink(x,a)
   res=sign( max(abs(x)-a,0));
end