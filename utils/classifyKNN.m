function [Y_pse] = classifyKNN(Xs,Ys,Xt,k)
   knn_model = fitcknn(Xs',Ys,'NumNeighbors',k);
   Y_pse = knn_model.predict(Xt');
end

