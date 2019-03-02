clear all

seed=0;
randn('state',seed); fprintf('seed %d\n', seed);
rand('state',seed); 

expnum = 30;
timelimit = [25;50]; 
testsize = [3;6];

[l, x]=read_sparse('rcv1-v2-train.svm');
classes=unique(l);
classes=classes((sum(repmat(l,1,size(classes,1))==repmat(classes',size(l,1),1),1)>=5)');
newdata=logical(sum(repmat(l,1,size(classes,1))==repmat(classes',size(l,1),1),2));
l=l(newdata);
x=x(newdata,:);

aobj=zeros(size(testsize,1),size(timelimit,1),expnum);
mobj=aobj; aprojnorm=aobj; mprojnorm=aobj;
annzW=aobj; annzH=aobj;

for prob=1:size(testsize,1)

  classnum=testsize(prob); r=classnum;

  perm=randperm(size(classes,1)); randclass=classes(perm(1:classnum));
  newdata=logical(sum(repmat(l,1,size(randclass,1))==repmat(randclass',size(l,1),1),2));
  V = x(newdata,:); 
  V = full(V(:,sum(V,1)>0)');     
%  V = sparse(V(:,sum(V,1)>0)'); 
  if sum((sum(V,1) == 0)) > 0,
    fprintf( 'Warning: empty documents\n');
  end
  n=size(V,1); m = size(V,2);
  fprintf('size %d %d %d nnz %d\n', n, r, m, nnz(V));

  for p=1:expnum,
    Winit = abs(randn(n,r));
    Hinit = abs(randn(r,m));

    fprintf( 'init condition number WtW %f HHt %f\n', cond(Winit'*Winit), cond(Hinit*Hinit'));
    for j=1:size(timelimit,1)
      [W,H] = nmf(V,Winit,Hinit,0.0000000000001,timelimit(j),8000);
      annzW(prob,j,p)=nnz(W); annzH(prob,j,p)=nnz(H); 
      obj = 0.5*norm(V-W*H,'fro')^2;    
      fprintf('alspgrad obj %f\n', obj);      
      aobj(prob,j,p)=obj;      
      gradW = W*(H*H') - V*H';
      gradH = (W'*W)*H - W'*V;
      aprojnorm(prob,j,p) = norm([gradW(gradW<0 | W>0); gradH(gradH<0 | H>0)]);
      fprintf( 'final condition number WtW %f HHt %f\n', cond(W'*W), cond(H*H'));
      
      [W1,H1,mresults] = mult(V,Winit,Hinit,0.0000000000001,timelimit(j),8000);
      obj = 0.5*norm(V-W1*H1,'fro')^2;    
      fprintf('mult obj %f \n', obj);      
      mobj(prob,j,p)=obj;
      gradW1 = W1*(H1*H1') - V*H1';
      gradH1 = (W1'*W1)*H1 - W1'*V;
      mprojnorm(prob,j,p) = norm([gradW1(gradW1<0 | W1>0.00001); gradH1(gradH1<0 | H1>0.00001)]);
    end
  end
end
for s=1:2,
  if s==1,
    resulta=mean(aobj,3); 
    for prob=1:size(testsize,1)
      for j=1:size(resulta,2)
	tmp=mobj(prob,j,:); 
	resultm(prob,j)=mean(tmp(~isnan(tmp))); 
      end
    end
    fprintf('Objective Value & \\mult'); 
  else
    resulta=mean(aprojnorm,3); 
    for prob=1:size(testsize,1)
      for j=1:size(resulta,2)
	tmp=mprojnorm(prob,j,:); 
	resultm(prob,j)=mean(tmp(~isnan(tmp))); 
      end
    end
    fprintf('\\\\ $\\|\\nabla f^P(W,H)\\|_F$ & \\mult'); 
  end
  minval=min(resulta,resultm);
  for prob=1:size(testsize,1)
    for j=1:size(resultm,2)
      if minval(prob,j) == resultm(prob,j)
	fprintf('& {\\bf %5.3f} ', resultm(prob,j));
      else
	fprintf('& %5.3f ', resultm(prob,j));
      end
    end
  end
  fprintf('\\\\ \n & \\alspgrad    '); 
  for prob=1:size(testsize,1),
    for j=1:size(resulta,2)
      if minval(prob,j) == resulta(prob,j)
	fprintf('& {\\bf %5.3f} ', resulta(prob,j));
      else
	fprintf('& %5.3f ', resulta(prob,j));
      end
    end
  end
  fprintf('\n');
end
% $$$   fprintf('\n annzW      '); fprintf('%10f ', mean(annzW,2));
% $$$   fprintf('\n annzH      '); fprintf('%10f ', mean(annzH,2));  
