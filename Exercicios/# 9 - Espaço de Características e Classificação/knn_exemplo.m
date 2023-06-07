function knn_exemplo()
clear
%amostras para teste
Teste=[2.2 4.4
       -4.2 2.3];
%amostras para treinamento
Treino=[ 1.0  2.0;  1.1  2.4;  1.3  2.1  %classe 1
        -4.0  5.0; -4.1  4.6; -4.2  4.2  %classe 2
        -3.0 -4.1; -3.1 -4.4; -3.3  -4.2]; %classe 3

hold on;
%plot das amostras de teste

plot(Teste(:,1) ,Teste(:,2), 'k^');
%plot das amostras de treinamento
plot(Treino(1:3,1) , Treino(1:3,2), 'go');%classe 1
plot(Treino(1:3,1) , Treino(1:3,2), 'rx');%classe 2
plot(Treino(4:6,1) , Treino(7:9,2), 'bd');%classe 2
axis([-6 5 -6 6]);

%grupos de cada uma das amostras de treinamento
Grupo=[1; 1; 1; 2; 2; 2; 3; 3; 3];
K=2;%número de viznhos mais próximos
%clusterização
Classes=knnclassify(Teste,Treino,Grupo,K);
disp(Classes);
