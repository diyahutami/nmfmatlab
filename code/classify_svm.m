clear

%load('../results/orldata.mat');
%load('../results/orlclass.mat');

load('../results/jaffedata-by_expression_32.mat');
load('../results/jaffeclass-by_expression_213.mat');

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
    t = templateSVM('Standardize',1);
    %fitcecoc : Fit multiclass models for support vector machines or other classifiers
    mdl = fitcecoc(X(train,:),Y(train),'Learners',t,...,
        'ClassNames',classOrder);
    predictions = predict(mdl,X(test,:));
    classperf(cp,predictions,test);
    result(i)=cp.CorrectRate
end

mean_result = mean(result)
stdev = std(result)

fprintf('mean %.5f stdev %.5f\n', mean_result, stdev);