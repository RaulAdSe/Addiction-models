% Compute the value of a MRP in 2 different ways.
% 10 states, reward only in the first one.
% Fix one value's state. You must have a reference.
clear all; clc;
N = 10;
%% Dynamic programming
tol = 1e-3;
it_max = 1e3;
it = 0;
Vs_old = zeros(1,N); %Vs_old(1) = 0.5;
eps = 1;
Rs = zeros(1,N); Rs(1) = 1; % Defines the rewards of going from state i to the next. No more than one option, no decisions.
next = [2:N, 1];  % Loop from states 2 to N, and then jump to state 1
rho = sum(Rs(:))/N;

% Preallocate the array to store first state values in each iteration
state_values = zeros(N, it_max);

while it < it_max
    for i = 1:N-1
        Vs_new(i) =  Rs(i) - rho + Vs_old(next(i));
    end
    Vs_new(N) = 0;    % fixed
    Vs_old = Vs_new;
    it = it + 1;
    
    % Store the first state value in each iteration
    state_values(:,it) = Vs_new;
    
    if it >= N + 1 && all(all(abs(diff(state_values(:,it-N:it),1, 2)) <= tol))
        break; % Exit the loop early if converged
    end
    
end

% Trim the preallocated arrays to remove unused zeros
state_values = state_values(:,1:it-1);

% Plot the convergence of the first state value
figure;
plot(1:length(state_values), state_values(:,:), 'o-');
xlabel('Iteration');
ylabel('Vs');
title('Convergence of V');
legend()

% Does not converge!
%% TDRL
alpha = 0.1;
beta = 0.2;
tol = 1e-3;
Vs_old = zeros(1,N);
r_bar_s = zeros(1,it_max); 
delta_s = zeros(1,it_max);
Vs_values = zeros(N, it_max);
it = 1;

while it < it_max
    if it <= N
        i = it;
    elseif mod(it, N) == 0
        i = N;
    else
        i = mod(it, N);
    end

    delta_s(it) = Rs(i) - r_bar_s(it) + Vs_old(next(i)) - Vs_old(i);
    Vs_new(i) = Vs_old(i) + alpha * delta_s(it);
    Vs_new(end) = 0; %fix
    r_bar_s(it + 1) = r_bar_s(it) + beta * delta_s(it);
    Vs_old = Vs_new;
    Vs_values(:,it) = Vs_new;  % Store Qs_new(1) value
    
    if it >= N + 1 && all(all(abs(diff(Vs_values(:,it-N:it),1, 2)) <= tol))
        break; % Exit the loop early if converged
    end
    
    it = it + 1;
end

% Trim the preallocated arrays to remove unused zeros
r_bar_s = r_bar_s(1:it-1);
delta_s = delta_s(1:it-1);
Vs_values = Vs_values(:,1:it-1);

% Plot the convergence of Qs_new(1)
figure;
subplot(3, 1, 1);
plot(1:length(Vs_values), Vs_values, 'o-');
xlabel('Iteration');
ylabel('Vs');
title('Convergence of Vs');
legend()
grid on;

% Plot the convergence of r_bar_s
subplot(3, 1, 2);
plot(1:length(r_bar_s), r_bar_s, 'o-');
xlabel('Iteration');
ylabel('Mean reward');
title('Convergence of r\_bar\_s');
grid on;

% Plot the convergence of delta_s
subplot(3, 1, 3);
plot(1:length(delta_s), delta_s, 'o-');
xlabel('Iteration');
ylabel('RPE');
title('Convergence of delta');
grid on;

%% Convergence study
% Define parameter ranges to sweep over
% It does not run properly the second time
N = 10;
it_max = 1e3;
alpha_values = linspace(0.01, 0.99, 10); % Generate 10 values between 0.01 and 0.99
beta_values = linspace(0.01, 0.99, 10);  % Generate 10 values between 0.01 and 0.99
% Initialize storage for convergence results
converged_combinations = zeros(length(alpha_values), length(beta_values));

% Loop through different combinations of alpha and beta values
for alpha_idx = 1:length(alpha_values)
    for beta_idx = 1:length(beta_values)
        alpha = alpha_values(alpha_idx);
        beta = beta_values(beta_idx);
        
        Vs_old = zeros(1,N);
        r_bar_s = zeros(1,it_max); 
        delta_s = zeros(1,it_max);
        Vs_values = zeros(N, it_max);
        it = 1;

        while it < it_max
            if it <= N
                i = it;
            elseif mod(it, N) == 0
                i = N;
            else
                i = mod(it, N);
            end

            delta_s(it) = Rs(i) - r_bar_s(it) + Vs_old(next(i)) - Vs_old(i);
            Vs_new(i) = Vs_old(i) + alpha * delta_s(it);
            Vs_new(end) = 0; %fix
            r_bar_s(it + 1) = r_bar_s(it) + beta * delta_s(it);
            Vs_old = Vs_new;
            Vs_values(:,it) = Vs_new;  % Store Qs_new(1) value

            if it >= N + 1 && all(all(abs(diff(Vs_values(:,it-N:it),1, 2)) <= tol))
                converged_combinations(alpha_idx, beta_idx) = 1; % Converged
                break; % Exit the loop early if converged
            end

            it = it + 1;
        end
    end
end
disp('Convergence matrix')
disp(converged_combinations)

% Plot the convergence matrix as a heatmap
figure;
imagesc(alpha_values, beta_values, converged_combinations);
colorbar
xlabel('Alpha value');
ylabel('Beta value');
title('Convergence Matrix');
