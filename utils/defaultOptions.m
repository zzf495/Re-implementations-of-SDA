function [options] = defaultOptions(varargin)
    options=varargin{1};
    n=nargin-1;
    if mod(n,2) ~=0
        error('Please enter coupled parameters\n');
        return ;
    end
    n=n/2;
    for i=1:n
       pos=1+2*i-1;
       key=varargin{pos};
       if ~isfield(options,key)
           val=varargin{pos+1};
           options=setfield(options,key,val);
       end
    end
end


