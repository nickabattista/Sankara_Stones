function Adv_Diff_PDE_1D()

%Author: Nicholas A. Battista
%University: UNC at Chapel Hill
%Date Created: August 4, 2014
%Last Modified: August 22, 2014

%Solves advjection-diffusion equations of the form: u_t + s*u_x = d*uxx + g(x)
%w/ u(0,x) = f(x)
%w/ s as wave-speed
%w/ d as diffusion coefficient

%1st: find projection of f(x) onto Chebyshev Basis of N Basis Elements, i.e., c(0)
%2nd: Fill matrices accordingly
%3rd: Find c(t) = e^{ T^-1 (-s Tp + d Tpp) t} * c(0)  [Homoegeneous Soln]
%4th: Find c_nh(t) = e^(At)*Integral[ exp(-At) T^-1 g(t,x) dt ] [Nonhomogeneous Soln]
%5th: Hold matrix info accordingly, t = [0 t1 t2 t3 ... ] and C = [c(0) c(t1) c(t2) .... ]

%It solves the problem: u_t + u_x = 0.1*u_xx + 0*g(x) 
%w/ initial value, u(0,x) = sin(pi*)
%and BCs, u(t,-1) = u(t,1) = 0

%NOTE: It can handle g=g(x) but solution becomes unstable later in
%simulation so for teaching purposes easier to just show 'pure'
%advection-diffusion

%NOTE: Case when the forcing function is g = g(t,x) and not just g = g(x) is
%still being worked on. (August 2014)

print_info();

t=0:0.025:1.0;            %Time for Simulation
sVec = [0.5 1.0 1.5];     %Wave Speed
dVec = [0.05 0.125 0.25]; %Diffusion Coefficient 
%NVec = [5 10 15];         %Number of Cheby. Functions/Pts.

NVec = [5 6 10];

fprintf('\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n\n');
fprintf('Parameter Sweep options\n');
fprintf('Type "1" for Wave Speed Sweep\n');
fprintf('Type "2" for Diffusion Coefficient Sweep\n');
fprintf('Type "3" for # of Chebyshev Pts. Sweep\n');

whichSim = input('Please choose what parameter sweep you would like to see: \n');

fprintf('\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n');

for n=1:3 %Begins parameter sweep loop.
    
    if n==1
        [d s N cj c_h c_nh] = give_Me_Parameters_and_Allocate(n,dVec,sVec,NVec,t,whichSim);
    else
        [d s N] = give_Me_Parameters(n,dVec,sVec,NVec,whichSim);
    end
  
    fprintf('\nPerforming Case: \n');
    fprintf('N=%d\n',N);
    fprintf('Running case: s = %d\n',s);
    fprintf('Running case: d = %d\n',d);

    %Compute Chebyshev collocation pts.
    x = collocation_points(N);
    
    %Compute Chebyshev coefficients for interpolating initial data, f(x)
    man = 1; %Manually Computed Coefficients Flag
    c0 = initial_coeffs(N,man);

    %Compute Matrices for Solving: T (d/dt) c = (-s Tp  + d Tpp) c
    T   = T_Mat(N,x);
    Tp =  Tp_Mat(N,x);
    Tpp = Tpp_Mat(N,x);
    
    %Computes Forcing Function Vector at Each Collocation Pt.
    gVec = g(N,x);
    
    %Matrix to be used in Matrix Exponential for Sol'n 
    A = T\( -s*Tp + d*Tpp );
    [V D] = eig(-A); %Finds e-vals and e-vectors of -A for inhomogeneous sol'n
    %if g = g(x) only
  
    intMat = inv(D); %Finds integration coefficients on exponentials, i.e., 1/lambda*exp(lambda*t)
    
    %Find spectral coefficients as function of time for each t_j
    for j=1:length(t)
       
       %homogeneous sol'n coefficients (j-which column to store, n-which simulation it's running)
       c_h(1:N+1,j,n) = expm( A*t(j) )*c0;
       
       %inhomogenous sol'n coefficients (j-which column to store, n-which simulation it's running)
       if j>1
            c_nh(1:N+1,j,n) = nonhomogeneous_coeffs(N,T,A,V,D,intMat,gVec,t(j)); %put in intMat if g=g(x) only 
       end
       
       %combines sol'n coefficients (j-which column to store, n-which simulation it's running)
       cj(1:N+1,j,n) = real(c_h(1:N+1,j,n) + c_nh(1:N+1,j,n));
       
    end %Ends finding spectral coefficiens for one choice of parameter
    
    fprintf('\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n');

end %Ends Parameter Sweep Loop

pause(1);

fprintf('\n\nLets plot the simulations now!\n');

pause(1);

plot_Simulations(N,cj,t,dVec,sVec,NVec,whichSim)

fprintf('\n\nWelp, thats a wrap!\n\n\n');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%Function to return parameter values for simulation and allocate storage
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [d s N cj c_h c_nh] = give_Me_Parameters_and_Allocate(n,dVec,sVec,NVec,t,whichSim)

    if whichSim == 1
        d = 0.125;       %Chooses which diffusion coefficient
        s = sVec(n);     %Chooses which wave speed
        N = 10;          %# of Spatial Pts (Chebyshev Pts.)
    elseif whichSim == 2
        d = dVec(n);     %Chooses which diffusion coefficient
        s = 1.0;         %Chooses which wave speed
        N = 10;          %# of Spatial Pts (Chebyshev Pts.)
    elseif whichSim == 3
        d = 0.125;       %Chooses which diffusion coefficient
        s = 1.0;         %Chooses which wave speed
        N = NVec(n);     %# of Spatial Pts (Chebyshev Pts.)
    else
        fprintf('\nTYPED SOMETHING WRONG! END SIMULATION! HURRRRRAYYYYY!\n\n');
        pause(2);
        fprintf('Please, restart.\n\n');
        pause(2);
        fprintf('Ending now...\n\n');
        pause(2);
        error('Simulation ended because of bad user input. Please be careful entering information.');
    end
    
    if n==1
        if  (whichSim ~= 3)
            %Allocate memory for storage of spectral coefficients
            cj=zeros(N+1,length(t),length(dVec));
            c_h = cj;
            c_nh = cj;
        else
            %Allocate memory for storage of spectral coefficients
            cj=zeros(NVec(3)+1,length(t),length(dVec));
            c_h = cj;
            c_nh = cj;  
        end
    end
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%Function to return parameter values for simulation only
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [d s N] = give_Me_Parameters(n,dVec,sVec,NVec,whichSim)

    if whichSim == 1
        d = 0.125;       %Chooses which diffusion coefficient
        s = sVec(n);     %Chooses which wave speed
        N = 10;          %# of Spatial Pts (Chebyshev Pts.)
    elseif whichSim == 2
        d = dVec(n);     %Chooses which diffusion coefficient
        s = 1.0;         %Chooses which wave speed
        N = 10;          %# of Spatial Pts (Chebyshev Pts.)
    elseif whichSim == 3
        d = 0.125;       %Chooses which diffusion coefficient
        s = 1.0;         %Chooses which wave speed
        N = NVec(n);     %# of Spatial Pts (Chebyshev Pts.)
    end
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%Function to return parameter values for simulation only
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [strTitle strLeg] = get_Plotting_Info(dVec,sVec,NVec,whichSim)

    if whichSim == 1
        d = 0.125;   %Chooses which diffusion coefficient
        N = 10;      %# of Spatial Pts (Chebyshev Pts.)
        
        
        leg1 = strcat('s = ',num2str(sVec(1)));
        leg2 = strcat('s = ',num2str(sVec(2)));
        leg3 = strcat('s = ',num2str(sVec(3)));
        strLeg = {leg1,leg2,leg3};
        
        dStr = num2str(d);
        NStr = num2str(N);
        strTitle = 'Wave Speed Sweep w/ ';
        str1 = strcat(' d = ',dStr);
        str2 = strcat(' N = ',NStr);
        strTitle = strcat(strTitle,str1,' and ',str2);
        
    elseif whichSim == 2
        s = 1.0;     %Chooses which diffusion coefficient
        N = 10;      %# of Spatial Pts (Chebyshev Pts.)
        
        leg1 = strcat('d = ',num2str(dVec(1)));
        leg2 = strcat('d = ',num2str(dVec(2)));
        leg3 = strcat('d = ',num2str(dVec(3)));
        strLeg = {leg1,leg2,leg3};
        
        sStr = num2str(s);
        NStr = num2str(N);
        strTitle = 'Diffusion Coefficient Sweep w/ ';
        str1 = strcat(' s = ',sStr);
        str2 = strcat(' N = ',NStr);
        strTitle = strcat(strTitle,str1,' and ',str2);
        
    elseif whichSim == 3
        d = 0.125; %Chooses which diffusion coefficient
        s = 1.0;   %Chooses which wave speed
        
        leg1 = strcat('N = ',num2str(NVec(1)));
        leg2 = strcat('N = ',num2str(NVec(2)));
        leg3 = strcat('N = ',num2str(NVec(3)));
        strLeg = {leg1,leg2,leg3};
        
        sStr = num2str(s);
        dStr = num2str(d);
        strTitle = '# of Chebyshev Polys Sweep w/ ';
        str1 = strcat(' s = ',sStr);
        str2 = strcat(' d = ',dStr);
        strTitle = strcat(strTitle,str1,' and ',str2);
        
    end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%Function that plots the simulations
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function plot_Simulations(N,cj,t,dVec,sVec,NVec,whichSim)

       [strTitle strLeg] = get_Plotting_Info(dVec,sVec,NVec,whichSim);

       leg1 = strLeg{1};
       leg2 = strLeg{2};
       leg3 = strLeg{3};
       
       xPts = -1:0.025:1;

       figure(1)
       for j=1:length(t)
            
            if whichSim < 3
                for i=1:length(xPts)
                    plot(xPts(i),f(xPts(i)),'o'); hold on;
                    plot(xPts(i),interpolate(N,xPts(i),cj(:,j,1)),'r*','MarkerSize',4); hold on;
                    plot(xPts(i),interpolate(N,xPts(i),cj(:,j,2)),'g*','MarkerSize',4); hold on;
                    plot(xPts(i),interpolate(N,xPts(i),cj(:,j,3)),'m*','MarkerSize',4); hold on;
                end
            else
                for i=1:length(xPts)
                    plot(xPts(i),f(xPts(i)),'o'); hold on;
                    plot(xPts(i),interpolate(NVec(1),xPts(i),cj(1:NVec(1)+1,j,1)),'r*','MarkerSize',4); hold on;
                    plot(xPts(i),interpolate(NVec(2),xPts(i),cj(1:NVec(2)+1,j,2)),'g*','MarkerSize',4); hold on;
                    plot(xPts(i),interpolate(NVec(3),xPts(i),cj(1:NVec(3)+1,j,3)),'m*','MarkerSize',4); hold on;
                end
            end
            axis([-1.2 1.2 -1.4 1.4]);
            title(strTitle);
            legend('initial condition',leg1,leg2,leg3,'Location','NorthWest');
            xlabel('x');
            ylabel('u(t,x)');
            pause(0.01);
            
            if j~=length(t)
            clf();
            end
       end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%Computes g(x) [the forcing function (inhomogeneity term) on RHS] at x
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function gVec = g(N,x)

gVec = zeros(N+1,1);
for i=1:N+1
   %gVec(i,1) = -0.005*(1-x(i)^2)*(1+x(i)^6);%0.1*cos(x(i));
   gVec(i,1) = 0;% sin(pi*x(i));
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%Computes f(x) [the initial value function] at x
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function val = f(x)

val = sin(pi*x); %Obeys BCs

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%Computes coefficients for inhomogeneous case using Variation of
%%Parameters. NOTE: Integrations have to be done by user and
%%hard-coded...nothing for free.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function val = nonhomogeneous_coeffs(N,T,A,V,D,intMat,gVec,t)

%Matrix of Exps: exp( lam*t_j ) on diag.
expDt = zeros(N+1,N+1);
for i=1:N+1
    a = D(i,i); %exponential coefficient
    expDt(i,i) = exp( a*t );
    %intMat(i,i) = ( a*sin(b*t) - b*cos(b*t) ) / (a^2+b^2);
 end

%val = expm(A*t)*V*intMat*expDt*inv(V)*inv(T)*gVec;

val = V*intMat*expDt*inv(V)*inv(T)*gVec;

val = expm(A*t)*(val - V*intMat*inv(V)*inv(T)*gVec);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%Finds c(0) coefficients for initial data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function c0 = initial_coeffs(N,man) 

%man - flag if want coefficients computed numerically via Gauss Quad. or
%just calling exact values that are stored in another script. Note: only
%stored for initial data of f(x) = sin(pi*x)

%Computes Initial Coefficients for Initial Condition
if man == 0
    c = zeros(N+1,1);
    for j=1:N+1
        F = @(x) f(x).*T(j-1,x)./sqrt(1-x.^2); 
        if j==1
            c(j) = 1/pi*quadl(F,-1,1,5e-10);
        else
            c(j) = 2/pi*quadl(F,-1,1,5e-10);    
        end
    end
else
    c = Initial_Data_Coeffs(N);
end
c0 = c;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%Finds N collocation points
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function val = collocation_points(N)

x = zeros(N+1,1);
for i=1:N+1
    x(N+2-i,1) = cos(pi*(i-1)/N);
end

val = x'; %% Need transpose

return

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%Builds Matrix of Second Derivatives, Tpp
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function jac = Tpp_Mat(N,x)

jac = zeros(N+1,N+1);

for k = 1:N+1  %%Runs over Collocation points on x axis

        for i = 1:N+1 %%Runs over ith basis function
                          
                if k == 1
                    jac(k,i) =  T(i-1,x(k));
                elseif k == N+1
                    jac(k,i) =  T(i-1,x(k));
                else
                    jac(k,i)  = Tpp(i-1,x(k));
                end
              
        end
end

return

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%Builds matrix of 1st Derivatives of Chebyshev Polys
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function val = Tp_Mat(N,x)               

for i=1:N+1     %Runs over Collocation pts. on x axis
    for j=1:N+1 %Runs over jth basis function
    
            if i==1
                val(i,j) = T(j-1,x(i));
            elseif i==N+1 
                val(i,j) = T(j-1,x(i));
            else
                val(i,j) = Tp(j-1,x(i));
            end
    end
end

return


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%Boundary Conditions [ie- Right hand Side of Lu = f]
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function val = T_Mat(N,x)               

for i=1:N+1     %Runs over Collocation pts. on x axis
    for j=1:N+1 %Runs over jth basis function
    
        %if i==1
        %    val(i,j)=  interpolate(N,x(i),y(j),un);
        %elseif i == N+1
        %    val(i,j)=  interpolate(N,x(i),y(j),un);
        %else
            val(i,j) = T(j-1,x(i));
        %end
      
    end
end

return



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%Series u(t,x) for a specific t value (i.e., feed in correct vector of
%%%coefficients)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function val = interpolate(N,x,un)               

val = 0;

for i=1:N+1 %Goes over all the N basis elemeents
         
      val = val + un(i)*T(i-1,x); 
       
end

%For computing polynomial interpolation using Standard Monomial Approach
%val2 = 0;
%for i=1:N+1
%    val2 = val2 + un(i)*x^(i-1);
%end

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
%%%T'(x)     (1st Derivative of Chebyshev Function); jth polynomial
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function val = Tp(j,x)             

if j==0
    val = 0;
else
    val = j*U(j-1,x);
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

fprintf('Author: Nicholas A. Battista\n');
fprintf('University: UNC at Chapel Hill\n');
fprintf('Date Created: August 4, 2014\n');
fprintf('Last Modified: August 22, 2014\n\n');

fprintf('Solves advjection-diffusion equations of the form: u_t + s*u_x = d*uxx + g(x)\n');
fprintf('w/ u(0,x) = f(x)\n');
fprintf('w/ s as wave-speed\n');
fprintf('w/ d as diffusion coefficient\n\n');

fprintf('IDEA:\n');
fprintf('1st: find projection of f(x) onto Chebyshev Basis of N Basis Elements, i.e., c(0)\n');
fprintf('2nd: Fill matrices accordingly\n');
fprintf('3rd: Find c_h(t) = e^{ T^-1 (-s Tp + d Tpp) t} * c(0)   [Homogeneous Soln]\n');
fprintf('4th: Find c_nh(t) = e^(At)*Integral[ exp(-At) T^-1 g(t,x) dt ]   [Nonhomogeneous Soln] \n');
fprintf('5th: Hold matrix info accordingly, t = [0 t1 t2 t3 ... ] and C = [c(0) c(t1) c(t2) .... ]\n\n');

fprintf('It solves the problem: u_t + u_x = 0.1*u_xx + 0*g(x)\n'); 
fprintf('w/ initial value, u(0,x) = sin(pi*)\n');
fprintf('and BCs, u(t,-1) = u(t,1) = 0\n\n');

fprintf('NOTE: It can handle g=g(x) but solution becomes unstable later in simulation so for teaching purposes easier to just show "pure" advection-diffusion\n\n');

fprintf('NOTE: Case when the forcing function is g = g(t,x) and not just g = g(x) is still being worked on. (August 2014)\n\n');

