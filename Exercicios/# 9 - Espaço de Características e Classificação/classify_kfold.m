function classify_kfold()
clear
load fisheriris;

%classes=[ones(1,50); 2*ones(1,50); 3*ones(1,50)]; %Três classes
classes=[ones(1,50); 2*ones(1,50)];%Duas classes
k=5;
indices=crossvalind('Kfold', 100,k);
vet_acuracia=zeros(1,k);

for i=1:k
    ind_train=find(indices~=i);
    ind_test=find(indices==1);
    base_train=meas(ind_train,:);
    base_test=meas(ind_test,:);
    classes_train=classes(ind_train);
    classes_test=classes(ind_test);
    class=classify(base_test, base_train, classes_train,'linear');
    
    vet_acuracia(i)=sum(class==classes_test)/length(class);

end
fprintf('Acurácia média - %f\n',mean(vet_acuracia));