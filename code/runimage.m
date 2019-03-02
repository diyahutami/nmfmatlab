% three image problems

clear all;

seed = 0;
randn('seed',seed);
rand('seed',seed);

probnum = 3;
expnum = 30;
timelimit = [25; 50]; 

aobj=zeros(probnum,size(timelimit,1),expnum);
mobj=aobj; 
aprojnorm=aobj; 
mprojnorm=aobj;

for prob=1:probnum
  switch prob,
   case 1, 
    V = cbcldata; r = 49;
   case 2,
    V = orldata; r = 25;
   case 3,
    V = natdata; r = 72;
  end
  fprintf('size %d %d %d nnz %d\n', size(V,1), r, size(V,2),nnz(V));
  
 for p=1:expnum,
    Winit=abs(randn(size(V,1),r)); 
    Hinit=abs(randn(r,size(V,2)));
    fprintf('init condition number WtW %f HHt %f\n', cond(Winit'*Winit), cond(Hinit*Hinit'));

    for j=1:size(timelimit,1)
      [W,H] = nmf(V,Winit,Hinit,0.0000000001,timelimit(j),8000);
      obj = 0.5*(norm(V-W*H,'fro')^2);    
      fprintf('alspgrad obj %f\n',obj);      
      aobj(prob,j,p)=obj;      
      gradW = W*(H*H') - V*H';
      gradH = (W'*W)*H - W'*V;
      aprojnorm(prob,j,p) = norm([gradW(gradW<0 | W>0); gradH(gradH<0 | H>0)]);
      fprintf('final condition number WtW %f HHt %f\n', cond(W'*W), cond(H*H'));
      
      [W1,H1,mresults] = mult(V,Winit,Hinit,0.0000000001,timelimit(j),8000);
      obj = 0.5*norm(V-W1*H1,'fro')^2;    
      fprintf('mult obj %f \n',obj);      
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
    resultm=mean(mobj,3);
    fprintf('Objective Value & \\mult'); 
  else
    resulta=mean(aprojnorm,3); 
    resultm=mean(mprojnorm,3);
    fprintf('\\\\ $\\|\\nabla f^P(W,H)\\|_F$ & \\mult     '); 
  end
  minval=min(resulta,resultm);
  for prob=1:probnum,
    for j=1:size(resultm,2)
      if minval(prob,j) == resultm(prob,j)
	fprintf('& {\\bf %5.2f} ', resultm(prob,j));
      else
	fprintf('& %5.2f ', resultm(prob,j));
      end
    end
  end
  fprintf('\\\\ \n & \\alspgrad '); 
  for prob=1:probnum,
    for j=1:size(resulta,2)
      if minval(prob,j) == resulta(prob,j)
	fprintf('& {\\bf %5.2f} ', resulta(prob,j));
      else
	fprintf('& %5.2f ', resulta(prob,j));
      end
    end
  end
  fprintf('\n');
end

