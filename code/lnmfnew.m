function [W,H,objhistory,iter,elapsed] = lnmfnew( V, rdim, alpha, beta, fname, showflag, stopconv, tol,timelimit, maxiter )
%

% Check that we have non-negative data
if min(V(:))<0, error('Negative values in data!'); end

% Globally rescale data to avoid potential overflow/underflow
V = V/max(V(:));

% Dimensions
vdim = size(V,1);
samples = size(V,2);

stopconv=stopconv; % stopping criterion (can be adjusted)
cons=zeros(samples,samples);
consold=cons;
inc=0;
j=0;

% Create initial matrices
W = abs(randn(vdim,rdim));
H = abs(randn(rdim,samples));

% Make sure W has unit sum columns!
W = W./(ones(vdim,1)*sum(W,1));

% Initialize displays
if showflag,
   figure(1); clf; % this will show the energies and sparsenesses
   figure(2); clf; % this will show the objective function
   drawnow;
end

% Calculate initial objective
objhistory = sum(sum((V.*log(V./(W*H))) - V + W*H)) + ...
	 alpha*sum(sum(W'*W)) - beta*sum(diag(H*H'));

timestarted = clock;
elapsed = etime(clock,timestarted);
% Start iteration
iter = 0;
for iter=1:maxiter,
    % stopping condition
    if objhistory(end) < tol | elapsed > timelimit,
        break;
    end

    % Show progress
    fprintf('[%d]: %.5f \n',iter,objhistory(end));    
    
    % Save every once in a while
    if rem(iter,5)==0,
        % test convergence every 5 iterations
        j=j+1;
        
        % construct connectivity matrix
        [y,index]=max(H,[],1);   %find largest factor
        mat1=repmat(index,samples,1);  % spread index down
        mat2=repmat(index',1,samples); % spread index right
        cons=mat1==mat2;
        
        if(sum(sum(cons~=consold))==0) % connectivity matrix has not changed
            inc=inc+1;                     %accumulate count 
        end
        fprintf('\t%d\t%d\t%d\n',iter,inc,sum(sum(cons~=consold))),
        
        if(inc>=stopconv)
            break,                % assume convergence is connectivity stops changing 
        end
        
        consold=cons;
        
        elapsed = etime(clock,timestarted);
        fprintf('Saving...');
        save(fname,'V','W','H','iter','objhistory','elapsed','inc');
        fprintf('Done!\n');
    end
    
    % Show stats
    if showflag & (rem(iter,5)==0),
	figure(1);
	cursW = (sqrt(vdim)-(sum(W)./sqrt(sum(W.^2))))/(sqrt(vdim)-1);
	cursH = (sqrt(samples)-(sum(H')./sqrt(sum(H'.^2))))/(sqrt(samples)-1);
	subplot(3,1,1); bar(sqrt(sum(W.^2)).*sqrt(sum(H'.^2)));
	subplot(3,1,2); bar(cursW);
	subplot(3,1,3); bar(cursH);
	if iter>1,
	    figure(2);
	    plot(objhistory(2:end));
	end
	drawnow;
    end
    
    % Update iteration count
    iter = iter+1;    
    
    % Save old values
    Wold = W;
    Hold = H;

    % Update rules (Matlab code by Stan Li) 
    VC = V./(W*H);
    H = sqrt(H.*(W'*VC));
    W = W.*(VC*H')./(repmat(sum(W,2),[1,rdim])+repmat(sum(H',1),[vdim,1]));
    W = W./(ones(vdim,1)*sum(W,1));
    
    % Calculate objective
    newobj = sum(sum((V.*log(V./(W*H))) - V + W*H)) + ...
	 alpha*sum(sum(W'*W)) - beta*sum(diag(H*H'));
    objhistory = [objhistory newobj];
    	    
    
end