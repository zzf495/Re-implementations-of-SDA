function [Sw] = withinScatter(X,Y)
    C=length(find(unique(Y)));
    Sw=0;
    for c=1:C
        Xc=X(:,Y==c);
        Xmean=mean(Xc,2);
        Fc=(Xc-Xmean);
        Sw=Sw+(Fc*Fc');
    end
end
