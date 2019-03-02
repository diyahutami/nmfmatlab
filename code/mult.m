function [W,H,iter] = mult(V,W0,H0,tol,timelimit,maxiter)

W = W0; H = H0;

initt=cputime;
t=initt;

for iter=1:maxiter,
  if (rem(iter,10)==0) | (iter==1),
    gradW = W*(H*H') - V*H';
    gradH = (W'*W)*H - W'*V;
    
    if iter==1,
      initgrad = norm([gradW; gradH'],'fro');
      fprintf('init grad norm %f\n', initgrad);
    end
    projnorm = norm([norm(gradW(gradW<0 | W>0.00001)); norm(gradH(gradH<0 | H>0.00001))]);
    if projnorm<tol*initgrad | iter == maxiter |  cputime-initt > ...
	  timelimit,
      fprintf('Iter = %d Final proj-grad norm %f\n', iter, projnorm);
      break
    end
  end
  W = W.*(V*H')./(W*(H*H'));
  H = H.*(W'*V)./((W'*W)*H);
end

