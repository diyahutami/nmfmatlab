function [ W,H ] = conCMF( V,k,num_experiment)
tic
% initt = cputime;
[m,n]=size(V);
loop=1;
while loop<=10
    if loop==1
        Hinit=complex(randn(k,n), randn(k,n));H = Hinit;
        Winit=complex(randn(n,k), randn(n,k));W=Winit;
        tol=0.1;
    else
        W=(pinv(V)*V)*pinv(H);
    end
    VtV = V'*V;
    beta = 0.1;theta=0.01;
    projgrad=1;
    iter=1;
    while projgrad>tol
        temp=projgrad;
        iter=iter+1;
        grad = -W'*VtV+W'*VtV*W*H;
        projgrad = norm(grad);
%        fprintf('conCMF select=%d, num_experiment=%d , k=%d, loop=%d, iter=%d and projgrad=%d \n',select,num_experiment,k,loop,iter,projgrad);
%        fprintf(' num_experiment=%d , k=%d, loop=%d, iter=%d and projgrad=%d \n',num_experiment,k,loop,iter,projgrad);
        % search step size
        t=1;
        for inner_iter=1:20,
            beta=beta^t;
            Hn = H - beta*grad;
            
            d = Hn-H;
            suff_decr =(0.5*norm(V-V*W*Hn,'fro')^2)-(0.5*norm(V-V*W*H,'fro')^2)-2*theta*sum(sum(real(grad).*real(d)+imag(grad).*imag(d)))<=0;
            
            if ~suff_decr
                t=t+1;
            else
                H=Hn;
                break;
            end
        end
        if iter>2000
            break;
        end
        if norm(temp-projgrad)<projgrad*10^-6
            iter=1000;
        end
%         if cputime-initt>3000
        if toc>10000
            break;
        end
    end
    loop=loop+1;
    if tol > 10^(-2)
        tol=tol/10;
    end
    if rem(loop,5)==0, fprintf('. '); end
    
end
fprintf('project gradient happened at projgrad=%d \n',projgrad);
end