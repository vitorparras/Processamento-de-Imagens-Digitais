function classify_holdout()
clear
load fisheriris;
[train,test]=crossvalind('HoldOut', 150,0.25);
classes=[ones(1,50); 2*ones(1,50); 3*ones(1,50)];
ind_train=find(train==1);
ind_test=find(test==1);
base_train=meas(ind_train,:);
base_test=meas(ind_test,:);
classes_train=classes(ind_train);
classes_test=classes(ind_test);
class=classify(base_test, base_train, classes_train,'linear');
acuracia=sum(class==classes_test)/length(class);
fprintf('Acurácia - %f\n',acuracia);
end