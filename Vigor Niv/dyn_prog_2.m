clear; close all;
% Fixing the mean reward
NT = 1e2;

nS = 2;
nA = 2;
nL = 1000;
beta = 1e2;

Q_0 = zeros(nS,nA);       % pairs of state action values
r_bar_0 = 0.1;

latencies = linspace(1,10,nL);   % Indexes and values are going to be the same
T = zeros(nS,nS,nA,nL);
Ur = zeros(nS,nA);
Ur(1,1) = 1;
reinforcement_schedule = zeros(nS,nA,nL);  % Define probability of reward for each (state, action, latency) triplet
reinforcement_schedule(1,1,:) = 1;
% Let's make it simple. No switching cost. Since we are only computing State 1.
Cu = zeros(nA);
Cv = zeros(nA);
Cv(1) = 0.5; Cv(2) = 0;
% No unit cost and no switching cost
r = zeros(NT, 1);
as = zeros(NT + 1, 1);
ls = zeros(NT + 1, 1);
delta = zeros(NT + 1, 1);
Q = zeros(nA,nS,nL);
Vs = zeros(NT + 1,1);
Qs = zeros(NT + 1,nA,nL);

as(1) = 2;
ss(1) = 2;
as(2) = 2;
ls(2) = 3;
ss(2) = 1;
storing = zeros(NT + 1,1);

tol = 1e-3;
it_max = NT;
it = 1;

r_bar_values = linspace(0,1,100); % Define different r_bar values
latency_indices = zeros(length(r_bar_values), 1); % Store latency indices for each r_bar
vals_max = zeros(length(r_bar_values), 1);
for r_bar_idx = 1:length(r_bar_values)
    r_bar_0 = r_bar_values(r_bar_idx);
    
    while it < it_max
        ss = 1;
        for a = 1:nA
            for l_i = 1:nL
                l = latencies(l_i);
                % Calculate the expected reward
                expected_reward = reinforcement_schedule(1, a, l_i) * Ur(1,a);
                % Calculate costs based on Cu and Cv
                cost = Cu(a) + Cv(a) / l + r_bar_0 * l;
                % Calulate next action value
                Q(a,1,l_i) = expected_reward - cost + 0;    % all states are all 0
            end
        end
        Qs(it,:,:) = Q(:,1,:);

        if it>3
            % Compute the element-wise difference
            diff = Qs(it,:,:) - Qs(it-1,:,:);
            % Compute the element-wise square of differences
            squared_diff = diff.^2;
            % Sum the squared differences element-wise
            sum_squared_diff = sum(squared_diff(:));
            % Take the square root to get the "norm"
            norm_metric = sqrt(sum_squared_diff);
        end

        if it>3 && norm_metric<tol
            break; % Exit the loop early if converged
        end
        it = it + 1;
    end

    Qs = Qs(1:it,:,:);
    
    % Find the index of the maximum Q value for State 1
    [vals, lat_index] = max(Qs(end,1,:));
    latency_indices(r_bar_idx) = lat_index;
    vals_max(r_bar_idx) = vals;
    % Reset variables for the next r_bar value
    it = 1;
    Qs = zeros(NT + 1,nA,nL);
end

% Create a plot of latency_indices vs. r_bar values
figure;
plot(r_bar_values, latencies(latency_indices), '-o');
xlabel('$\bar{R}^*$', 'Interpreter', 'latex'); % Add a bar over "R" using LaTeX syntax
ylabel('Maximizing latencies');

figure;
plot(r_bar_values, vals_max, '-o');
xlabel('$\bar{R}^*$', 'Interpreter', 'latex'); % Add a bar over "R" using LaTeX syntax
ylabel('V(S=1)');
