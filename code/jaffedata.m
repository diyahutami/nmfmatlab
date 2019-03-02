function V = jaffedata
% jaffedata - read face image data from jaffe database
%

global imloadfunc;
    
% This is where the cbcl face images reside
thepath = '../data/jaffe/by_expression_256/';
fname = '../results/jaffedata-by_expression_256.mat';

% Create the data matrix
V = zeros(256*256,213);

% Read the directory listing
D = dir(thepath);

% Step through each image, reading it into the data matrix
% Note: The (+2) is just to avoid '.' and '..' entries
fprintf('Reading in the images...\n');
imloadfunc = ''
for i=1:213,
    switch imloadfunc,
     case 'pgma_read',
      I = pgma_read([thepath D(i+2).name]);
     otherwise,
      I = imread([thepath D(i+2).name]);
    end
    V(:,i) = reshape(I,[256*256, 1]);
    if rem(i,100)==1, fprintf('[%d/24]',floor(i/100)); end
end

fprintf('\n');

% Same preprocessing as Lee and Seung
V = V - mean(V(:));
V = V / sqrt(mean(V(:).^2));
V = V + 0.25;
V = V * 0.25;
V = min(V,1);
V = max(V,0);   


% Additionally, this is required to avoid having any exact zeros:
% (divergence objective cannot handle them!)
V = max(V,1e-4);


fprintf('Saving...');
save(fname,'V');
fprintf('Done!\n');

