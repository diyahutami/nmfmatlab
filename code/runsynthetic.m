clear all;

global glbitera glbiterW glbiterH glbnumt 

testsize=[30 5 30; 80 20 80; 130 30 130];

tol = [0.001; 0.0001; 0.00001];
seed=0;
% matlab
randn('state',seed); 
% octave
% randn("seed",seed); 
fprintf('seed %d\n', seed);

expnum=30;

% commented out if using octave
aiterW=zeros(size(testsize,1),size(tol,1),expnum);
aiterH=aiterW;
 
for i=1:size(testsize,1),
  
  % variables to store results
  atime=zeros(size(tol,1),expnum);
  mtime=atime; ptime=atime;
  aobj=atime;  mobj=atime;  mobj2=atime;    
  pobj=atime;  
  aiter=atime; miter=atime; piter=atime;
  anumt=atime;
  atime2=zeros(expnum,1); aobj2=atime2; aiter2=atime2;
  
  n=testsize(i,1);  r=testsize(i,2);  m=testsize(i,3);

  for p=1:expnum, 
    V = abs(randn(n,m));
    Winit = abs(randn(n,r));
    Hinit = abs(randn(r,m));

    if i==2 & p == 1,
      save testconv V Winit Hinit
    end
    for j=1:size(tol,1)
      t=cputime;
      [W,H] = nmf(V,Winit,Hinit,tol(j),1000,8000);
      t1=cputime -t; atime(j,p)=t1; aiter(j,p)=glbitera;
      aiterW(i,j,p)=glbiterW; aiterH(i,j,p)=glbiterH; 
      anumt(j,p)=glbnumt; 
      fprintf('alspgrad time %f\n',t1);
      aobj(j,p)=0.5*(norm(V-W*H,'fro')^2);    
    
      t=cputime;
      [W1,H1,miter(j,p)] = mult(V,Winit,Hinit,tol(j),1000,8000);
      t1=cputime -t; mtime(j,p)=t1; 
      fprintf('mult time %f\n',t1);
      mobj(j,p)=0.5*(norm(V-W1*H1,'fro')^2);    
      
      t=cputime;
      [W2,H2,piter(j,p)] = pgrad(V,Winit,Hinit,tol(j),1000,8000);
      t1=cputime -t; ptime(j,p)=t1; 
      fprintf('pgrad time %f\n',t1);
      pobj(j,p)=0.5*sum(sum((V-W2*H2).^2));
    end
    
    % additional test for alspgrad
    t=cputime;
    [W,H] = nmf(V,Winit,Hinit,tol(size(tol,1))/10,1000,8000);
    t1=cputime -t; atime2(p)=t1; aobj2(p)=0.5*sum(sum((V-W*H).^2)); aiter2(p)=glbitera;
    fprintf('alspgrad time %f\n',t1);

% $$$     if i==2,
% $$$       [W1,H1,mresults] = mult(V,Winit,Hinit,tol(size(tol,1)),1000,50000);
% $$$       mobj2(1,p)=0.5*sum(sum((V-W1*H1).^2));	
% $$$     end
    
  end
  fprintf(' tol        '); fprintf(1,'%10.5f ', tol); 
  fprintf('\n mobj2       '); fprintf('%10f ', mean(mobj2,2));
  fprintf('\n anumt      '); fprintf('%10f ', mean(anumt,2));
  fprintf('\n');
  resultt=[mean(mtime,2) mean(atime,2) mean(ptime,2)];
  resulti=[mean(miter,2) mean(aiter,2) mean(piter,2)];
  resulto=[mean(mobj,2) mean(aobj,2) mean(pobj,2)];

  for p=1:3,
    mint(p)=min(resultt(p,:)); mino(p)=min(resulto(p,:));
  end
  for p=1:3,
    switch p
     case 1
      fprintf('\n \\mult    '); 
     case 2
      fprintf('\\\\ \n \\alspgrad');
     case 3
      fprintf('\\\\ \n \\pgrad   ');      
    end
    for j=1:size(tol,1),
      if mint(j)==resultt(j,p)
	fprintf('& {\\bf %5.2f}', resultt(j,p)); 
      else
	fprintf('& %5.2f', resultt(j,p)); 
      end
    end
    if p == 2,
      fprintf('& {\\bf %5.2f}', mean(atime2));
    else
      fprintf('&      ');
    end
    for j=1:size(tol,1),
	fprintf('& %5.0f', resulti(j,p)); 
    end
    if p == 2,
      fprintf('& %5.0f', mean(aiter2));
    else
      fprintf('&      ');
    end
    prevo = -inf;
    for j=1:size(tol,1),    
      if j>1, 
	prevo = resulto(j-1,p);
      end
      if mino(j)==resulto(j,p) 
        if j > 1 & resulto(j,p) >= prevo,
	    fprintf('& $^*${\\bf %5.2f}', resulto(j,p));
	else
	  fprintf('& {\\bf %5.2f}', resulto(j,p)); 
	end
      else
        if j > 1 & resulto(j,p) >= prevo,
	  fprintf('& %5.2f$^*$', resulto(j,p)); 
	else
	  fprintf('& %5.2f', resulto(j,p)); 	  
	end
      end
    end
    if p == 2,
      fprintf('& {\\bf %5.2f}', mean(aobj2));
    else
      fprintf('&      ');
    end
  end
  fprintf('\n');
end
fprintf('\n$W$'); 
for i=1:size(testsize,1)
  fprintf('& %5.2f ', mean(aiterW(i,:,:),3));
end
fprintf('\\\\ \n $H$');   
for i=1:size(testsize,1)
  fprintf('& %5.2f ', mean(aiterH(i,:,:),3));
end
fprintf('\n');
