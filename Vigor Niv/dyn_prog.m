% clear; close all;
% Looks for the best latency and mean reward because of it.

NT = 1e6;

nS = 2;
nA = 2;
nL = 100;

Q_0 = zeros(nS,nA);       % pairs of state action values
r_bar_0 = 0;

latencies = linspace(0.1,5,nL);   % Indexes and values are going to be the same
T = zeros(nS,nS,nA,nL);
Ur = zeros(nS,nA);
Ur(1,1) = 1;
reinforcement_schedule = zeros(nS,nA,nL);  % Define probability of reward for each (state, action, latency) triplet
reinforcement_schedule(1,1,:) = 1;
% Let's make it simple. No switching cost. Since we are only computing State 1.
Cu = zeros(nA);
Cv = zeros(nA);
Cv(1) = 0.2; Cv(2) = 1;
% No unit cost and no switching cost
r = zeros(NT, 1);
r_bar = zeros(NT + 1, 1);
r_bar(1) = r_bar_0;
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
time_spent = zeros(NT + 1,1);

tol = 1e-6;
it_max = NT;
it = 2;

while it < it_max
    ss = 1;
    for a = 1:nA
        for l_i = 1:nL
            l = latencies(l_i);
            % Calculate the expected reward
            expected_reward = reinforcement_schedule(1, a, l_i) * Ur(1,a);
            % Calculate costs based on Cu and Cv
            cost = Cu(a) + Cv(a) / l + r_bar(it) * l;
            % Calulate next action value
            Q(a,1,l_i) = expected_reward - cost + 0;    % all states are all 0
        end
    end
    % Choose the action that has maximum Q value
    [val_max,tau_max_indx] = max(Q(1,1,:));
    storing(it) = storing(it-1) + (reinforcement_schedule(1, 1, tau_max_indx) * Ur(1,1) - (Cu(1) + Cv(1) / latencies(tau_max_indx)))/latencies(tau_max_indx);
    time_spent(it) = time_spent(it-1) + latencies(tau_max_indx);
    r_bar(it+1) = storing(it)/time_spent(it);
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
    
    if it>3 && norm_metric<tol && abs(r_bar(it) - r_bar(it-1)) < tol
        break; % Exit the loop early if converged
    end
    it = it + 1;
end

Qs = Qs(1:it,:,:);
r_bar = r_bar(1:it,:,:);



% Create a single figure with all subplots
% figure;
% for action = 1:nA
%     for latency = 1:nL
%         subplot(nA, nL, (action - 1) * nL + latency);  % Create a subplot grid
%         plot(squeeze(Qs(1:end-1, action, latency)));
%         title(['Action ', num2str(action), ', Latency ', num2str(latency)]);
%         xlabel('Time Step');
%         ylabel('Q-Value');
%         ylim([0 1]);  % Set y-axis limits to 0 and 1
%     end
% end
%         

[max_val, lat_index] = max(Qs(end,1,:));
figure()
for l_i = 1:length(latencies)
    scatter(latencies(l_i),Qs(end,1,l_i))
    hold on
end
line([latencies(lat_index), latencies(lat_index)], ylim, 'Color', 'r', 'LineStyle', '--');
xlabel('Latencies');
ylabel('Q*-Value of State 1');

figure()
plot(r_bar)
xlabel('Time steps');
ylabel('$\bar{R}^*$', 'Interpreter', 'latex'); % Add a bar over "R" using LaTeX syntax

