fname = ['../results/result_jaffe_by_expression_patch_100_nmf_sc_svm_100_w03_h03_2.mat'];
% Read in the results
load(fname);

% Print iteration
fprintf('Iter: %d\n',iter);

% DISPLAY COLUMNS OF W
%figure(1); 
%fprintf('%d \n', ceil(sqrt(size(H',2))))
%visual(H',1,ceil(sqrt(size(H',2))),56);

% SHOW OBJECTIVE FUNCTION HISTORY
figure(2); 
if exist('objhistory'),
    if length(objhistory)>1,
        plot(objhistory(2:end));
    else
        clf;
    end
end

% ANALYZE SPARSENESS
vdim = size(W,1);
samples = size(H,2);
  
figure(3);
% How much is each unit utilized (in terms of total energy)?
subplot(3,1,1); bar(sqrt(sum(W.^2)).*sqrt(sum(H'.^2)));  
% How sparse are the basis vectors?
cursW = (sqrt(vdim)-(sum(abs(W))./sqrt(sum(W.^2))))/(sqrt(vdim)-1);
subplot(3,1,2); bar(cursW);
% How sparse are the coefficients
cursH = (sqrt(samples)-(sum(abs(H'))./sqrt(sum(H'.^2))))/(sqrt(samples)-1);
subplot(3,1,3); bar(cursH);

% PRINT OUT FINAL APPROXIMATION ERROR

WH = W*H; 
Emse = 0.5*sum(sum((V-WH).^2));
V(find(V<eps))=eps; 
WH(find(WH<eps))=eps;
Ediv = sum(sum((V.*log(V./(WH))) - V + WH));
  
fprintf('---------------------------------\n');
fprintf('Approximation error:\n');
fprintf('MSE = %.5f \n',Emse);
fprintf('Div = %.5f \n',Ediv);
fprintf('---------------------------------\n');
  
% SHOW HOW MUCH TIME IT TOOK...
  
fprintf('Running time (in seconds):\n');
fprintf('%.1f\n',elapsed);
fprintf('---------------------------------\n');  