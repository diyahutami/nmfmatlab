clear

load('../results/orldata.mat');
load('../results/orlclass.mat');

X = V'
Y = V1
classOrder = unique(Y);
rng(1); % For reproducibility

kfold=10;
indices = crossvalind('Kfold',Y,kfold,'Classes',classOrder);

cp = classperf(Y);

result = []
for i = 1:10
    test = (indices == i); 
    train = ~test;
    mdl = fitcknn(X(train,:),Y(train),'NumNeighbors',1);
    predictions = predict(mdl,X(test,:));
    classperf(cp,predictions,test);
    result(i)=cp.CorrectRate
end

mean = mean(result)
stdev = std(result)

fprintf('mean %.5f stdev %.5f\n', mean, stdev);