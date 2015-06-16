function Nonlinear_Polar_2D


% Author: Nicholas A. Battista
% University: Rochester Institute of Technology
% Created: Octoer 2009
% Last Revision: September 2010
%
%This is a 2D non-linear pseudo-spectral elliptic PDE solver in polar coordinates
%
%Running the code solves the following non-linear polar elliptic PDE:
%
%     Laplacian(u) + u^2 = f(r,theta)\n
%with
%     f(r,theta) = 4*u_rr + (2/r)*u_r + ((1/(2r))^2)*u_thetatheta + u^2,
%with Dirichelet/Neumann BCs in r and periodic BCs in theta,
%     u_r(0,theta)=0, u(1,theta)=0, u(r,theta)=u(r,theta+2*pi),
%and exact solution,
%     u(r,theta) = (cos((2*r-1)*pi) + 1)*exp(cos(theta)).
%

print_info();

Start_Num = 2;
End_Num = 8;

for NN=Start_Num:End_Num
    
    N = 2*NN;       %%# of Chebyshev Collocation Points
    M = N;          %%# of Fourier Collocation Points (MUST BE EVEN)

    %For first iteration, constructs the vectors containing the errors.
    if NN==Start_Num
        NerrorL2 = zeros(1,N);
        NerrorInf = NerrorL2;
        time = zeros(1,N);
    end
    
    %Finds solution for particular number of basis functions, N
    [A theta un NerrorL2 NerrorInf time] = find_Solution(N,M,NerrorL2,NerrorInf,time);

end %ends for loop at beginning looping over number of grid pts

fprintf('\n -------------------------------------------------------------- \n\n');

plot_collocation_grid(N,M,A,theta);

plot_solution(N,M,un);

plot_error_convergence(Start_Num,End_Num,NerrorL2,NerrorInf);

plot_time_increase(Start_Num,End_Num,time);

fprintf('\n\nThat is it! Thanks!\n\n');



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [A theta un NerrorL2 NerrorInf time] = find_Solution(N,M,NerrorL2,NerrorInf,time)

%collocation points in r and theta directions
A = cheby_collocation_points(N);                    %%Cheby collocation pts b/w [-1,1]
theta = fourier_collocation_points(M);              %%Fourier collocation pts b/w [-pi,pi]

%initial guess for spectral coefficients
un = initial_guess(N,M);

tol = 1e-8;            %Error Tolerance for Newton's Method
err= 1;                %Error to initialize Newton's Method
n=1;                   %Counter

fprintf('\n -------------------------------------------------------------- \n\n');
fprintf('%d, %d (# of basis functions in r and theta)\n\n',N,M);
fprintf('NEWTON METHOD\n');
fprintf('Step | Error\n');
tic
while err > tol
    
    J = jacobian(N,M,A,theta,un);
    fn = build_rhs(N,M,A,theta,un);
    un1 = un - J\fn;
    err = sqrt((un1-un)'*(un1-un));
    un = un1;
    
    fprintf('  %d  | %d\n',n,err);
    n=n+1;
end
time(N) = toc;

fprintf('Newton Method Converged within tol of %d\n\n',tol);

[NerrorL2 NerrorInf] = expconv(N,M,un,NerrorL2,NerrorInf);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [NerrorL2 NerrorInf] = expconv(N,M,un,NerrorL2,NerrorInf)

%Grids to compare solution over
rrr = 0:.025:1;
ttt = -pi:.05:pi;

%Allocating memory for exact-soln' (each sep. of var. component seperately)
urexact = ones(length(rrr),1);
u_theta_exact = ones(length(ttt),1);

%Computes each separation on variable solution respectively
for i = 1:length(rrr)    
    urexact(i) = cos((2*rrr(i)-1)*pi)+1;
end
for j = 1:length(ttt)
    u_theta_exact(j) = exp(cos(ttt(j)));   
end


%%%Creates matrix of exact solution @ points [0,1]x[-pi,pi] in steps of 0.01
%%%Creates spectral solution @ points [0,1]x[-pi,pi] in steps of 0.01
%%%Finds difference between exact and spectral solution
uexact = ones(length(rrr),length(ttt));
sol = uexact;
error = uexact;
for i = 1:length(rrr)
    for j = 1:length(ttt)
        uexact(i,j) = urexact(i)*u_theta_exact(j);
        sol(i,j) = interpolateR(N,M,rrr(i),ttt(j),un);
        error(i,j) = (sol(i,j) - uexact(i,j))^2;        
    end
end

%Computes L2-Norm Error
absError = sqrt(sum(sum(error)));
fprintf('The L2-Norm Error is: %d\n',absError);

%Computes Inf-Norm Error
maxError = max(max(abs(error)));
fprintf('The Inf-Norm Error is: %d\n',maxError);

%L2-Norm Error Vector
NerrorL2(N) = absError;

%Inf-Norm Error Vector
NerrorInf(N) = maxError;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function val = cheby_collocation_points(N)

x = zeros(1,N+1);
for i=1:N+1
    x(N+2-i) = cos(pi*(i-1)/N);
end
val = x';     %% Need transpose
return


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function val = fourier_collocation_points(M)

theta = zeros(1,M);
for k=1:M
    theta(k) = 2*pi*k/M;
end
val = theta'; %% Need transpose
return

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function f = Fourier_Eval_Matrix(M,theta)

f = zeros(M,M);
for k = 1:M       %Runs over collocation pts
    for l = 1:M   %Runs over Fourier Series
        n = l - M/2;
        f(k,l) = exp(1i*n*theta(k));
    end
end

return

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function plot_collocation_grid(N,M,A,theta)

fprintf('\nplotting collocation grid...\n');

figure(1)
subplot(1,2,1)
for k = 1:N+1
    for j = 1:M-1
        plot(A(k),theta(j),'r*','MarkerSize',10); hold on;
    end
end
xlabel('A')
ylabel('phi')
title('Collocation Points')

subplot(1,2,2)
for k = 1:N+1
    for j = 1:M-1
        polar(theta(j),A(k),'r*'); hold on;
    end
end
xlabel('x')
ylabel('y')
title('Collocation Points')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function val = initial_guess(N,M)

untmp = zeros(1,(N+1)*(M-1));
for i = 1:(N+1)*(M-1)
    untmp(i) = 0;
end

val = untmp';

return

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%Builds Jacobian Matrix
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function jac = jacobian(N,M,A,theta,un)

fourier = Fourier_Eval_Matrix(M,theta);

for k = 1:N+1                          %%Runs over Chebyshev Collocation points
    for l = 1:M-1                      %%Runs over Fourier Collocation points
        
        row = (k-1)*(M-1)+(l-1+1);
        
        for m = 1:N+1                  %%Runs over ith Chebyshev Polynomial
              
            TT = T(m-1,A(k));
            TTp = Tp(m-1,A(k));
            TTpp = Tpp(m-1,A(k));
            
            for j = 1:M-1               %%Runs over nth Fourier Sine Series Function
                
                n = j - M/2;
                col = (m-1)*(M-1) + (j-1+1);
                
                if k == 1
                    jac(row,col) = TTp*fourier(l,j);%TTp*exp(i*n*theta(l));
                elseif k == N+1
                    jac(row,col) = TT*fourier(l,j);%TT*exp(i*n*theta(l));%
                else
                    jac(row,col) = ( 4*TTpp + (4/(1+A(k)))*TTp - (2*n/(1+A(k)))^2*TT )*fourier(l,j) + 2*interpolateA(N,M,A(k),theta(l),un)*TT*fourier(l,j);
                end  
           end
       end
    end
end

return


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%Builds Right Hand Side (Boundary Conditions) [ie- Right hand Side of Lu = f]
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function val = build_rhs(N,M,A,theta,un)               

for k=1:N+1 %Runs over Chebyshev Collocation pts. on x axis
    
    uR_rr =  -pi^2*cos(A(k)*pi);
    uR_r  =  -pi*sin(A(k)*pi);
    uR    =  cos(A(k)*pi) + 1 ;
    
    for l=1:M-1 %Runs over Fourier Collocation pts on y axis
        
        row = (k-1)*(M-1)+(l-1+1);
       
        urr =  uR_rr*exp(cos(theta(l)));
        ur =   uR_r  *exp(cos(theta(l)));
        u_thetatheta = (-cos(theta(l)) + (sin(theta(l)))^2 )*exp(cos(theta(l)))*uR;
        u = uR*exp(cos(theta(l)));
        
        if k==1
            rhs(row)= interpolateTpA(N,M,A(k),theta(l),un);  
        elseif k == N+1
            rhs(row)= interpolateA(N,M,A(k),theta(l),un); 
        else
            rhs(row) = ( useriesPDE(N,M,A(k),theta(l),un) + (interpolateA(N,M,A(k),theta(l),un))^2 - ( 4*urr + (4/(1+A(k)))*ur + ((2/(1+A(k)))^2)*u_thetatheta + u^2 ) ); 
        end
    end
end

val = rhs';

return

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Series u_{rr} + 1/r*u_{r} + u_thetatheta
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function val = useriesPDE(N,M,A,t,un)

val = 0;

for m=1:N+1
    
     TT = T(m-1,A);
     TTp = Tp(m-1,A);
     TTpp = Tpp(m-1,A);
    
   for j=1:M-1
        
      row = (m-1)*(M-1)+(j-1+1);
      n = j - M/2;
      
      val = val +  un(row)* ( ( 4*TTpp + (4/(1+A))*TTp - (2*n/(1+A))^2*TT )*exp(i*n*t) ) ;
       
   end
end

return

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%Series u(x,y)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function val = interpolateA(N,M,A,theta,un)               

val = 0;

for m =1:N+1
    
    TT = T(m-1,A);
    
    for j=1:M-1
      
      k = j - M/2;
      row = (m-1)*(M-1)+(j-1+1); 
      
      a(row) = real(un(row));      %%COSINE COEFFICIENTS
     
      if k < 0
          b(row) = imag(un(row));  %%SINE COEFFICIENTS
      elseif k == 0
          b(row) = 0;              %%SINE COEFFICIENTS
      else
          b(row) = imag(un(row));  %%SINE COEFFICIENTS 
      end

      val = val + ( a(row)*cos(k*theta) + b(row)*sin(k*theta) )*TT;
      
    end
end

return

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%Series u(x,y)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function val = interpolateR(N,M,r,theta,un)               

val = 0;

for m =1:N+1
    
    TT = T(m-1,2*r-1);
    
    for j=1:M-1
      
      k = j - M/2;
      row = (m-1)*(M-1)+(j-1+1); 
      
      a(row) = real(un(row));      %%COSINE COEFFICIENTS
     
      if k < 0
          b(row) = imag(un(row));  %%SINE COEFFICIENTS
      elseif k == 0
          b(row) = 0;             %%SINE COEFFICIENTS
      else
          b(row) = imag(un(row));  %%SINE COEFFICIENTS 
      end

      val = val + ( a(row)*cos(k*theta) + b(row)*sin(k*theta) )*TT;
      
    end
end

return



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%Series u(x,y)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function val = interpolateTpA(N,M,A,theta,un)               

val = 0;

for m =1:N+1
    
    TT = Tp(m-1,A);
    
    for j=1:M-1
      
      k = j - M/2;
      row = (m-1)*(M-1)+(j-1+1); 
      
      a(row) = real(un(row));      %%COSINE COEFFICIENTS
     
      if k < 0
          b(row) = imag(un(row));  %%SINE COEFFICIENTS
      elseif k == 0
          b(row) = 0;             %%SINE COEFFICIENTS
      else
          b(row) = imag(un(row));  %%SINE COEFFICIENTS 
      end

      val = val + ( a(row)*cos(k*theta) + b(row)*sin(k*theta) )*TT;
      
    end
end

return


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%T''(x)  (2nd Derivative of Chebyshev Function); jth Cheby. polynomial
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function val = Tpp(j,x)           

if x == 1
    val = (j^4-j^2)/3;
elseif x == -1
    val = (-1)^j*(j^4-j^2)/3;
else
    val = (j*(j+1)*T(j,x)-j*U(j,x))/(x^2-1);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%T'(x)  (1st Derivative of Chebyshev Function); jth Cheby. polynomial
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function val = Tp(j,x)           

val = j*U(j-1,x);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%T(x)     (Chebyshev Function); jth polynomial
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function val = T(j,x)             

if x == 1
    val = 1;
elseif x == -1
    val = (-1)^j;
else
    val = cos(j*acos(x));
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%U(x)    (2nd Chebyshev Function); jth 2nd Cheby. polynomial
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function val = U(j,x)            

if x == 1
    val = j+1;
elseif x == -1
    val = (-1)^j*(j+1);
elseif j == -1
    val = 0;
else
    val = sin((j+1)*acos(x))/sin(acos(x));
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function plot_solution(N,M,un)

rr = 0:.05:1;
tt = -pi:.05:pi;

fprintf('\nplotting exact solution vs. numerical solution...\n');

umatrix = zeros(length(tt),length(rr));
for i = 1:length(rr)
    for j=1:length(tt)
        umatrix(j,i) = interpolateR(N,M,rr(i),tt(j),un);
    end
end

figure(2)
subplot(1,2,1)
mesh(rr,tt,umatrix)
xlabel('r')
ylabel('phi')
zlabel('u(r,phi)')
title('Numerical Solution')

subplot(1,2,2)
ezsurf('(cos((2*x-1)*pi) + 1)*exp(cos(y))',[0 1 -pi pi])
title('Exact Solution')
xlabel('r')
ylabel('phi')
zlabel('u(r,phi)')


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function plot_error_convergence(S,E,NerrorL2_Bef,NerrorInf_Bef)


fprintf('\nplotting error convergence...\n');

%Resizing vectors for plotting
cL2 = 1;  %Index counter for L2-Error Vec.
cInf = 1; %Index counter for Inf-Error Vec.
for i=1:length(NerrorL2_Bef)
    
    if NerrorL2_Bef(i) > 0
        NerrorL2(cL2) = NerrorL2_Bef(i);
        cL2 = cL2+1;
    end
    
    if NerrorInf_Bef(i) > 0
        NerrorInf(cInf) = NerrorInf_Bef(i);
        cInf = cInf + 1;
    end 
end

count = 2*S:2:2*E; %Since needs to be even for sol'n


figure(3)
%
subplot(2,2,1)
plot(count,NerrorL2,'*');
xlabel('N')
ylabel('L2-Error')
title('Error Convergence: L2-Error vs. N')
%
subplot(2,2,2)
semilogy(count,NerrorL2,'*');
xlabel('N')
ylabel('Log(L2-Error)')
title('Error Convergence: Log(L2-Error) vs. N')
%
subplot(2,2,3)
plot(count,NerrorInf,'*');
xlabel('N')
ylabel('Inf-Norm Error')
title('Error Convergence: Inf-Norm Error vs. N')
%
subplot(2,2,4)
semilogy(count,NerrorInf,'*');
xlabel('N')
ylabel('Log(Inf-Norm Error)')
title('Error Convergence: Log(Inf-Norm Error) vs. N')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function plot_time_increase(S,E,time_bef)

fprintf('\nplotting time complexity...\n');

count = 2*S:2:2*E;

%Resizing vectors for plotting
c = 1;  %Index counter for time vec.
for i=1:length(time_bef)
    if time_bef(i) > 0
        time(c) = time_bef(i);
        c = c + 1;
    end 
end

figure(4)
subplot(1,2,1);
plot(count,time,'*');
xlabel('N (and M)')
ylabel('Time for Each Simulation')
title('Time Complexity vs. N (and M)')
%
subplot(1,2,2);
semilogy(count,time,'*');
xlabel('N (and M)')
ylabel('Log(Time for Each Simulation)')
title('Log(Time Complexity) vs. N (and M)')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function print_info()

fprintf('\n\nThis is a 2D non-linear pseudo-spectral elliptic PDE solver in polar coordinates\n');
fprintf('Author: Nicholas A. Battista \n');
fprintf('Last update: September 2010\n\n');
fprintf('Running the code solves the following non-linear polar elliptic PDE:\n\n');

fprintf('     Laplacian(u) + u^2 = f(x,y)\n\n');
fprintf('with\n');
fprintf('     f(x,y) = 4*u_rr + (2/r)*u_r + ((1/(2r))^2)*u_thetatheta + u^2,\n\n');
fprintf('with Dirichelet/Neumann BCs in r and periodic BCs in theta,\n');
fprintf('     u_r(0,theta)=0, u(1,theta)=0, u(r,theta)=u(r,theta+2*pi),\n\n');
fprintf('and exact solution,\n');
fprintf('     u(r,theta) = (cos((2*r-1)*pi) + 1)*exp(cos(theta)).\n');
fprintf('\nNote: This simulation will take roughly 10 minutes to complete the convergence study\n');
fprintf('\n -------------------------------------------------------------- \n');
fprintf('\n  -->> BEGIN THE SIMULATION <<--\n');

