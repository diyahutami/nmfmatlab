clear

%load('../data/ALLAML.mat');

%load('../results/orldata.mat');
%load('../results/orlclass.mat');

load('../results/jaffedata-by_expression_patch_100_32.mat');
load('../results/jaffeclass-by_expression_213.mat');

% Random number generator seed
seed = 0;
randn('seed',seed);
rand('seed',seed);

V = V'
Y = V1

%V = D'
%Y = classes01

classOrder = unique(Y);

result = []

for iter=1:10
    kfold=10;
    indices = crossvalind('Kfold',Y,kfold,'Classes',classOrder);

    cp = classperf(Y);

    rdim = 100;
    sW = 0.5;
    sH = 0.5;
    %fname = ['../results/result_orl_nmf_sc_svm_25_w07_h07.mat'];
    %fname = ['../results/result_allaml_nmf_sc_svm_100_w03_h03.mat'];
    fname = ['../results/result_jaffe_by_expression_patch_100_nmf_sc_svm_100_w05_h05.mat'];
    showflag = 0;
    tol = 0.00001
    stopconv = 30
    timelimit = 100
    maxiter = 5000
    [W,H,objhistory,iter,elapsed] = nmfscnew( V, rdim, sW, sH, fname, showflag, stopconv, tol,timelimit, maxiter )

    result_cross = []
    for i = 1:10
        test = (indices == i); 
        train = ~test;
        t = templateSVM('Standardize',1);
        %fitcecoc : Fit multiclass models for support vector machines or other classifiers
        mdl = fitcecoc(W(train,:),Y(train),'Learners',t,...,
            'ClassNames',classOrder);
        predictions = predict(mdl,W(test,:));
        classperf(cp,predictions,test);
        result_cross(i)=cp.CorrectRate
    end
    result=[result result_cross]
end

mean_result = mean(result)
stdev = std(result)

fresult = ['../results/result_jaffe_by_expression_patch_100_nmf_sc_svm_100_w05_h05_classify.mat'];
fprintf('Saving...');
save(fresult,'result','mean_result','stdev');
fprintf('mean %.5f stdev %.5f\n', mean_result, stdev);