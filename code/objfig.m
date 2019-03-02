clear all

n = 80; r = 20; m = 80;
load testconv

times=0.5; timee=1024;
i=1; time=times;
while time <= timee,
  [W1,H1] = mult(V,Winit,Hinit,0.000000001,time,10000000);
  objm(i) = 0.5*(norm(V-W1*H1,'fro')^2);    
  timem(i)=time;
  i = i + 1;
  time = time * 2;
end

times=0.5; timee=32;
i=1; time=times;
while time <= timee,
  [W,H] = nmf(V,Winit,Hinit,0.000000001,time,1000000);
  obja(i) = 0.5*(norm(V-W*H,'fro')^2);    
  timea(i) = time;
  i = i + 1;
  time = time * 2;
end

semilogx(timea, obja, '-', timem, objm, '--')
set(gca, 'fontsize', 18) ; 
set(findobj('Type', 'line'), 'LineWidth', 3)  
xlabel('Time in seconds (logged scale)'); ylabel('Objective value');
print -deps testconv.eps
