function [acc] = getAcc(Ypse,Y)
    if size(Ypse,2)>1
        Ypse=Ypse';
    end
     if size(Y,2)>1
        Y=Y';
     end
    acc=length(find(Ypse==Y))/length(Y);
end

