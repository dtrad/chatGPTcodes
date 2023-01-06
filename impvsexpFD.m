
    
% Set up the grid
L = 1; % length of the domain
N = 100; % number of grid points
x = linspace(0, L, N); % grid points
h = x(2) - x(1); % grid spacing

% Set the time step
dt = 0.001;

% Set the initial conditions
u0 = sin(2*pi*x); % initial temperature

% Solve the heat equation using the explicit method
u_explicit = u0;
t = 0;
while t < 0.1
    u_explicit = u_explicit + dt * heatEqn(t, u_explicit, x, h);
    t = t + dt;
end

% Solve the heat equation using the implicit method
u_implicit = u0;
t = 0;
while t < 0.1
    u_implicit = (u_implicit + dt * heatEqn(t+dt, u_implicit, x, h)) / (1 + dt*(1/h^2));
    t = t + dt;
end

% Plot the solutions
% plot(x, u_explicit, x, 0*u_implicit);
subplot(121),plot(x, u_explicit),title('explicit')
subplot(122),plot(x, u_implicit),title('implicit')

xlabel('x');
ylabel('u');
%legend('Explicit', 'Implicit');

% Define the heat equation as a function
function du = heatEqn(t, u, x, h)
    N = length(x);
    du = zeros(N, 1);
    du(1) = (u(2) - u(1)) / h^2;
    du(2:N-1) = (u(3:N) - 2*u(2:N-1) + u(1:N-2)) / h^2;
    du(N) = (u(N) - u(N-1)) / h^2;
end
    
    
    
    