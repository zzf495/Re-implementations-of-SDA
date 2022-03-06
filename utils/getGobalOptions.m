function [options,list] = getGobalOptions(varargin)
    % Create time: 21-06-06 by 495
    % Description: This function is aim to generate grid-search objects by
    % two ways: key-value and value-only. The examples are shown below:
    % input: key-value
    %       getGobalOptions('alpha',[1,2,3],'beta',2)
    % output: 
    %       options: [
    %               {'alpha':1,'beta':2},
    %               {'alpha':2,'beta':2},
    %               {'alpha':3,'beta':2}
    %       ]
    % input: value-only
    %       getGobalOptions([1,2,3],2)
    % output: 
    %       options: [
    %               {1:1,2:2},
    %               {1:2,2:2},
    %               {1:3,2:2}
    %       ]
    %---------------------------------------
    %list=[  1 2
    %        2 2
    %        3 2  ]
    parameterNumber=nargin;
    step=1;
    if nargin ==0
        error('At least one parameter is required\n');
        return ;
    end
    if isstr(varargin{1})
        if mod(nargin,2) ~=0
            error('Please enter coupled parameters\n');
            return ;
        else
            parameterNumber=int8(nargin/2);
            step=2;
        end
    end
   
   
    list=[];
    num=[]; % prod(num) 数组求乘积
    for i=1:parameterNumber
       num=[num,length(varargin{i*step})];
    end
    options=[];
    ind=fullfact(num);
    for i=1:parameterNumber
        list=[list,reshape(varargin{i*step}(ind(:,i)),[],1)];
    end
    for i=1:length(list(:,1))
        options_child=struct();
        for k=1:parameterNumber
            if step==2
                options_child=setfield(options_child,varargin{k*step-1},list(i,k));
            else
                options_child{k}=list(i,k);
            end
        end
        %---------------------kernel--------------
        if isfield(options_child,'kernel_type')
            if options_child.kernel_type==0
                options_child.kernel_type='primal';
            elseif options_child.kernel_type==1
                options_child.kernel_type='linear';
            elseif options_child.kernel_type==2
                options_child.kernel_type='rbf';
            end
        end
        options=[options;options_child];
    end
end


