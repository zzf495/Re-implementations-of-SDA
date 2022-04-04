function [result] = getMinMax(Vec,k,maxmin,precision)
% Get k-th max/min value of V
%% input:
%%% Vec: n*1, the list of values
%%% k: integer, the k-th value
%%% maxmin: 'max': the k-th largest value , 'min' the k-th lowest value
%%% precision: the precision
%% output:
%%% result: interger, the k-th largest/lowest value
    if nargin==3
        precision=0;
    end
    if strcmpi(maxmin,'max')
        func=@max;
        fillVal=-inf;
    elseif strcmpi(maxmin,'min')
        func=@min;
        fillVal=inf;
    else
        error('error input: maxmin!');
    end
    for i=1:k
        [result,~]=func(Vec);
        Vec( (Vec>=result-precision) & (Vec<=result+precision))=fillVal;
    end
end

