function Nonlinear_2D 

% Author: Nicholas A. Battista
% University: Rochester Institute of Technology
% First Created: October 2008
% Last Revision: September 2010
%
%This is a 2D non-linear elliptic, pseudo-spectral PDE solver
%It assumes a solution of the form: Sum_i Sum_j c_ij T_i(x) T_j(y)
%
%It solves the problem:
%                   Laplacian(u) + u^2 = f(x,y)
%with
%  f(x,y) = -pi^2 ( cos(pi*x)sin(pi*y) + sin(pi*y)( cos(pi*x) + 1 ) ) + (sin(pi*y)( cos(pi*x) + 1 ))^2
%
%and BCs
%                   u(-1,y)=u(1,y)=u(x,-1)=u(x,1)=0
%
%with the exact solution,
%                   u(x,y) = sin(pi*y)( cos(pi*x) + 1 )
%


Start_Num = 3;  %Number of collocation pts to start simulation
End_Num =  12;  %Number of collocation pts to end simulation

print_info();

for N=Start_Num:End_Num
    
    %For first iteration, constructs the vectors containing the errors.
    if N==Start_Num
        NerrorL2 = zeros(1,N);
        NerrorInf = NerrorL2;
        time = zeros(1,N);
    end
        
    %Finds solution for particular number of basis functions, N
    [x y un NerrorL2 NerrorInf time] = find_Solution(N,NerrorL2,NerrorInf,time);

end %ends for loop at beginning looping over number of grid pts

plot_collocation_grid(N,x,y);

plot_solution(N,un)

plot_error_convergence(Start_Num,End_Num,NerrorL2,NerrorInf);

plot_time_increase(Start_Num,End_Num,time);

fprintf('\n\nThat is it! Thanks!\n\n');




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [x y un NerrorL2 NerrorInf time] = find_Solution(N,NerrorL2,NerrorInf,time)

%Computes collocation points in x and y directions
x = collocation_points(N);
y = x;

un = initial_guess(N); %Initial guess for spectral coefficients
tol = 1e-8;            %Error Tolerance for Newton's Method
err= 1;                %Error to initialize Newton's Method
n=1;                   %Counter


fprintf('\n -------------------------------------------------------------- \n\n');
fprintf('%d (# of basis functions in x and y)\n\n',N);

%Stores Function,Deriv, and 2nd Deriv. Cheby. Poly Values
[Tval T_pp] = all_Cheby(x);

fprintf('NEWTON METHOD\n');
fprintf('Step | Error\n');
tic
while err > tol
    
    J = jacobian(N,x,y,un,Tval,T_pp);
    fn = build_rhs(N,x,y,un,Tval,T_pp);
    un1 = un - J\fn;
    err = sqrt((un1-un)'*(un1-un));
    un = un1;
    
    fprintf('  %d  | %d\n',n,err);
    n=n+1;
end
time(N) = toc;

fprintf('Newton Method Converged within tol of %d\n\n',tol);

[NerrorL2 NerrorInf] = expconv(N,un,NerrorL2,NerrorInf);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function val = initial_guess(N)

untmp = zeros(1,(N+1)^2);

val = untmp';

return


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function plot_solution(N,un)

fprintf('\nplotting exact solution vs. numerical solution\n');

xx = -1:.025:1;
yy = -1:.025:1;

len = length(xx);
umatrix = zeros(len,len);

for i = 1:len
    for j=1:len
        umatrix(j,i) = interpolate(N,xx(i),yy(j),un);
    end
end

figure(2)
subplot(1,2,1)
mesh(xx,yy,umatrix)
xlabel('x')
ylabel('y')
zlabel('u(x,y)')
title('Numerical Solution')
%
subplot(1,2,2)
ezsurf('(cos(pi*x) + 1)*(sin(pi*y))',[-1 1 -1 1])
title('Exact Solution')
zlabel('u(x,y)')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [NerrorL2 NerrorInf] = expconv(N,un,NerrorL2,NerrorInf)

%Grids to compare solution over
xxx = -1:.025:1;
yyy = -1:.025:1;

oneVector = ones(length(xxx),1);
uxexact = oneVector;
uyexact = oneVector;

%Computes each separation on variable solution respectively
for i = 1:length(xxx)
    uxexact(i,1) = cos(xxx(i)*pi)+1;   
    uyexact(i,1) = sin(yyy(i)*pi)  ;   
end


%%%Creates matrix of exact solution @ points [-1,1]x[-1,1] in steps of 0.01
%%%Creates spectral solution @ points [-1,1]x[-1,1] in steps of 0.01
%%%Finds difference between exact and spectral solution
uexact = ones(length(xxx),length(xxx));
sol = uexact;
error = uexact;
for i = 1:length(xxx)
    for j = 1:length(xxx)
        uexact(i,j) = uxexact(i)*uyexact(j);
        sol(i,j) = interpolate(N,xxx(i),yyy(j),un);
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


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function plot_error_convergence(S,E,NerrorL2,NerrorInf)

fprintf('\nplotting error convergence...\n');

count = 1:1:E;

figure(3)
%
subplot(2,2,1)
plot(count(S:E),NerrorL2(S:E),'*');
xlabel('N')
ylabel('L2-Error')
title('Error Convergence: L2-Error vs. N')
%
subplot(2,2,2)
semilogy(count(S:E),NerrorL2(S:E),'*');
xlabel('N')
ylabel('Log(L2-Error)')
title('Error Convergence: Log(L2-Error) vs. N')
%
subplot(2,2,3)
plot(count(S:E),NerrorInf(S:E),'*');
xlabel('N')
ylabel('Inf-Norm Error')
title('Error Convergence: Inf-Norm Error vs. N')
%
subplot(2,2,4)
semilogy(count(S:E),NerrorInf(S:E),'*');
xlabel('N')
ylabel('Log(Inf-Norm Error)')
title('Error Convergence: Log(Inf-Norm Error) vs. N')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function plot_time_increase(S,E,time)

fprintf('\nplotting time complexity...\n');

count = 1:1:E;

figure(4)
subplot(1,2,1);
plot(count(S:E),time(S:E),'*');
xlabel('N')
ylabel('Time for Each Simulation')
title('Time Complexity vs. N')
%
subplot(1,2,2);
semilogy(count(S:E),time(S:E),'*');
xlabel('N')
ylabel('Log(Time for Each Simulation)')
title('Log(Time Complexity) vs. N')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function val = collocation_points(N)

%global N
for i=1:N+1
    x(N+2-i) = cos(pi*(i-1)/N);
end

val = x'; %% Need transpose

return

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function plot_collocation_grid(N,x,y)

fprintf('\nplotting collocation grid...\n');

figure(1)
for i =1:N+1
    for j = 1:N+1
        plot(x(i),y(j),'r*','MarkerSize',10); hold on;
    end
end
plot([-1 -1],[-1 1]); hold on;
plot([1 1],[-1 1]); hold on;
plot([-1 1],[1 1]); hold on;
plot([-1 1],[-1 -1]); hold on;
title('Collocation Grid')
xlabel('x-collocation pts');
ylabel('y-collocation pts');
axis([-1.1 1.1 -1.1 1.1]);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%Builds Jacobian Matrix
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function jac = jacobian(N,x,y,un,Tval,T_pp)

for k = 1:N+1  %%Runs over Collocation points on x axis
    for l = 1:N+1  %%Runs over Collocation points on y axis
        
        row = (k-1)*(N+1)+(l-1+1);
        
        for i = 1:N+1 %%Runs over Collocation points on y axis
           for j = 1:N+1 %%Runs over gth Cheby. Polynomial
          
                col = (i-1)*(N+1) + (j-1+1);
                
                if k == 1
                    jac(row,col) = Tval(i,k)*Tval(j,l);%T(i-1,x(k))*T(j-1,y(l));
                elseif k == N+1
                    jac(row,col) = Tval(i,k)*Tval(j,l);%T(i-1,x(k))*T(j-1,y(l));
                elseif l==1
                    jac(row,col) = Tval(i,k)*Tval(j,l);%T(i-1,x(k))*T(j-1,y(l));
                elseif l == N+1
                    jac(row,col) = Tval(i,k)*Tval(j,l);%T(i-1,x(k))*T(j-1,y(l));
                else
                    jac(row,col)  = T_pp(i,k)*Tval(j,l) + T_pp(j,l)*Tval(i,k) + 2*interpolate(N,x(k),y(l),un)*Tval(i,k)*Tval(j,l);
                    %jac(row,col)  = Tpp(i-1,x(k))*T(j-1,y(l)) + Tpp(j-1,y(l))*T(i-1,x(k)) + 2*interpolate(N,x(k),y(l),un)*T(i-1,x(k))*T(j-1,y(l));
                end
              
           end
       end
    end
end

return


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%Boundary Conditions [ie- Right hand Side of Lu = f]
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function val = build_rhs(N,x,y,un,Tx,Tx_pp)               

for i=1:N+1 %Runs over Collocation pts. on x axis
    for j=1:N+1 %Runs over Collocation pts on y axis
        
        row = (i-1)*(N+1)+(j-1+1);
        if i==1
            rhs(row)=  interpolate(N,x(i),y(j),un);
        elseif i == N+1
            rhs(row)=  interpolate(N,x(i),y(j),un);
        elseif j==1
            rhs(row) = interpolate(N,x(i),y(j),un);
        elseif j == N+1
            rhs(row) = interpolate(N,x(i),y(j),un);
        else
            rhs(row) = useriespp(N,x(i),y(j),un) + (interpolate(N,x(i),y(j),un))^2 + pi^2 * ( cos(pi*x(i)) * sin(pi*y(j)) + sin(pi*y(j))*(cos(pi*x(i)) + 1 ) ) - (sin(pi*y(j)))^2*( cos(pi*x(i))+1)^2;
        end
    end
end

val = rhs';

return


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Series u_{xx}(x,y) + u_{yy}(x,y)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function val = useriespp(N,x,y,un)

%global N un
val = 0;

for i=1:N+1
   for j=1:N+1
        
      row = (i-1)*(N+1)+(j-1+1); 
      val = val + un(row)* ( Tpp(i-1,x)*T(j-1,y) + T(i-1,x)*Tpp(j-1,y) ) ;
       
   end
end

return


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%Series u(x,y)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function val = interpolate(N,x,y,un)               

%global N un 
val = 0;

for i=1:N+1
   for j=1:N+1
        
      row = (i-1)*(N+1)+(j-1+1); 
      val = val + un(row)*T(i-1,x)*T(j-1,y); 
       
   end
end

return

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%Stores all Function, Derivative, and Second Derivative Vals of Cheby
%%%Polys for speedup
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [TA TA_pp] = all_Cheby(A)

for a=1:length(A)
    for k=1:length(A)
        TA(k,a) = T(k-1,A(a));
        TA_pp(k,a) = Tpp(k-1,A(a));
        
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
else
    val = sin((j+1)*acos(x))/sin(acos(x));
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function print_info()

fprintf('\n\nThis is a 2D non-linear pseudo-spectral elliptic PDE solver\n');
fprintf('Author: Nicholas A. Battista \n');
fprintf('Last update: September 2010\n\n');
fprintf('Running the code solves the following non-linear elliptic PDE:\n\n');

fprintf('     Laplacian(u) + u^2 = f(x,y)\n\n');
fprintf('with\n');
fprintf('     f(x,y) = -pi^2*( cos(pi*x)*sin(pi*y) + sin(pi*y)*(cos(pi*x)+1) ) + ( sin(pi*y)*cos(pi*x) )^2,\n\n');
fprintf('with Dirichelet BCs,\n');
fprintf('     u(-1,y)=u(1,y)=u(x,-1)=u(x,1)=0,\n\n');
fprintf('and exact solution,\n');
fprintf('     u(x,y) = sin(pi*y)( cos(pi*x) + 1 ).\n');
fprintf('\n Note: This simulation will take roughly 12 minutes to complete the convergence study\n');
fprintf('\n -------------------------------------------------------------- \n');
fprintf('\n  -->> BEGIN THE SIMULATION <<--\n');



