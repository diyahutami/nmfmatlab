function [W,H,iter] = pgrad(V,W0,H0,tol,timelimit,maxiter)

W = W0; H=H0;
iter=1; 
initt=cputime;

alpha = 1;
for iter=1:maxiter,
  gradW = W*(H*H') - V*H';
  gradH = (W'*W)*H - W'*V;
  if (iter==1)
    initgrad=norm([gradW; gradH'],'fro');    
    fprintf('init grad norm %f\n', initgrad);
    H = nlssubprob(V,W,H0,0.001,1000);    
    obj = 0.5*(norm(V-W*H,'fro')^2);    
    continue;
  end
  
  projnorm = norm([gradW(gradW<0 | W>0); gradH(gradH<0 | H>0)]);  
  if projnorm<tol*initgrad | cputime-initt > timelimit,
    fprintf('final grad norm %f\n', projnorm);
    break
  end
  Wn = max(W - alpha*gradW,0);    
  Hn = max(H - alpha*gradH,0);    
  newobj = 0.5*(norm(V-Wn*Hn,'fro')^2);
  if newobj-obj > 0.01*(sum(sum(gradW.*(Wn-W)))+ ...
    	                sum(sum(gradH.*(Hn-H)))),
    % decrease alpha    
    while 1,
      alpha = alpha/10;
      Wn = max(W - alpha*gradW,0);    
      Hn = max(H - alpha*gradH,0);    
      newobj = 0.5*(norm(V-Wn*Hn,'fro')^2);
      if newobj - obj <= 0.01*(sum(sum(gradW.*(Wn-W)))+ ...
    	                       sum(sum(gradH.*(Hn-H)))),
        W = Wn; H = Hn;
        obj = newobj;
        break;
      end
    end
  else 
    % increase alpha
    while 1,
      Wp = Wn; Hp = Hn; objp = newobj;
      alpha = alpha*10;
      Wn = max(W - alpha*gradW,0);    
      Hn = max(H - alpha*gradH,0);    
      newobj = 0.5*(norm(V-Wn*Hn,'fro')^2);
      if newobj - obj > 0.01*(sum(sum(gradW.*(Wn-W)))+ ...
    	                      sum(sum(gradH.*(Hn-H)))) | (Wn==Wp & Hn==Hp),
        W = Wp; H = Hp;
        obj = objp; alpha = alpha/10;
        break;
      end
    end
  end
end
