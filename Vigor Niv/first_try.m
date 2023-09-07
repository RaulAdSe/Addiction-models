clear; close all;

NT = 5e6;
nS = 2;
nA = 2;
nL = 50;

beta_initial = 0;  % Initial low beta
beta_final = 100;    % Target beta
anneal_steps = round(NT-2); % Number of steps to anneal beta

Q_0 = zeros(nS,nA);       % pairs of state action values
r_bar_0 = 0;

latencies = [0.1:0.1:5];   % Indexes and values are going to be the same
T = zeros(nS,nS,nA,nL);
Ur = zeros(nS,nA);
Ur(1,1) = 1;
reinforcement_schedule = zeros(nS,nA,nL);  % Define probability of reward for each (state, action, latency) triplet
reinforcement_schedule(1,1,:) = 1;
Cu = zeros(nA);
Cv = zeros(nA);
Cu(1) = 0.3; Cu(2) = 0.2; 
Cv(1) = 0.3; Cv(2) = 0;

% No unit cost and no switching cost

ss = zeros(NT + 1, 1);
ss(1) = 1;
r = zeros(NT, 1);
r_bar = zeros(NT + 1, 1);
r_bar(1) = r_bar_0;
as = zeros(NT + 1, 1);
ls = zeros(NT + 1, 1);
delta = zeros(NT + 1, 1);
Q = zeros(nA,nS,nL);
Vs = zeros(NT + 1,1);
Qs = zeros(NT + 1,nA,nL);

drug_effectiveness = 1.0;  % Initial drug effectiveness
tolerance_rate = 0;     % Rate of tolerance development (adjust as needed)
craving = 0;  % Initial craving (can adjust the initial value)
craving_increase_rate = 0;
drug_decay_rate = 1e-6;  % Adjust this rate to control the drug's decay speed

as(1) = 2;
ss(1) = 2;
as(2) = 2;
ls(2) = 3;
ss(2) = 1;
storing = zeros(NT + 1,1);
time_spent = zeros(NT + 1,1);
drug = zeros(NT + 1,1);


for t = 2:NT
    
    % Calculate current beta using linear annealing
    current_beta = beta_initial + (beta_final - beta_initial) * min(t / anneal_steps, 1);

    % Compute the softmax logits
    logits = Q(:,ss(t),:).* current_beta;
    max_logit = max(logits(:));  % Find the maximum logit value

    % Compute the softmax probabilities using the log-sum-exp trick
    ps = exp(logits - max_logit);  % Subtracting max_logit for numerical stability
    ps = ps ./ sum(ps(:));

    % Select action according to softmax
    action_idx = find(rand() < cumsum(ps(:)), 1);  % Find the index where the cumulative sum exceeds a random value

    % Convert the action index back to (action, latency) pair
    [action, latency_idx] = ind2sub([nA, nL], action_idx);
    as(t) = action;
    ls(t) = latencies(latency_idx);
    
    if ss(t) == 2
        ls(t) = 0;
    end
    
    % States manual change
    if ss(t) == 1
        ss(t+1) = 2;
    else
        ss(t+1) = 1;
    end
    % Calculate the action value
    if ss(t) == 1
        if as(t) == 1
            % Update drug effectiveness to reflect tolerance
            drug_effectiveness = max(0.01, drug_effectiveness - tolerance_rate);
            craving = craving + craving_increase_rate;
            expected_reward = reinforcement_schedule(ss(t), as(t), latency_idx) * Ur(as(t)) * drug_effectiveness * (1 + craving);
            drug(t) = drug(t-1)+1;
        else 
            expected_reward = reinforcement_schedule(ss(t), as(t), latency_idx) * Ur(as(t));
            drug(t) = drug(t-1);
        end
        % Calculate the expected reward
        % Calculate costs based on Cu and Cv
        cost = Cu(as(t)) + Cv(as(t)) / ls(t) + r_bar(t) * ls(t);
        Q(as(t),ss(t),latency_idx) = expected_reward - cost + 0;
        storing(t) = storing(t-1) + Ur(ss(t),as(t)) - Cu(as(t)) + Cv(as(t)) / ls(t);
        delta(t) = expected_reward + 0 - Q(as(t),ss(t),latency_idx) -r_bar(t);
    else
        Q(:,2,:) = 0;
        storing(t) = storing(t-1);
        Q_state_1 = Q(:,1,:);
        Vs(t) = max(Q_state_1(:));
        delta(t) = 0 + Vs(t) - 0 -r_bar(t);
        drug(t) = drug(t-1);
    end
    
    drug(t) = drug(t) * (1-drug_decay_rate * (1 + ls(t)));
    time_spent(t) = time_spent(t-1) + ls(t);
    r_bar(t+1) = storing(t)/time_spent(t);
    Qs(t,:,:) = Q(:,1,:);
    
end

% Initialize variables to store action frequencies
action1_freq = zeros(1, ceil(NT/1e4));
action2_freq = zeros(1, ceil(NT/1e4));

% Create indices to separate data every 10,000 time steps
split_indices = 1:1e4:NT;
split_indices(end+1) = NT;  % Add NT as the last split index

% Iterate over each 10,000 time step segment
for i = 1:length(split_indices)-1
    start_idx = split_indices(i);
    end_idx = split_indices(i+1)-1;
    
    % Extract the relevant data for this segment
    segment_ss = ss(start_idx:end_idx);
    segment_as = as(start_idx:end_idx);
    
    % Calculate action frequencies in state 1
    action1_freq(i) = sum(segment_ss == 1 & segment_as == 1);
    action2_freq(i) = sum(segment_ss == 1 & segment_as == 2);
end

% Create a time vector for the x-axis
time_steps = 1e4:1e4:NT;

% Create a bar plot to visualize the evolution of action frequencies
figure;
bar(time_steps, [action1_freq; action2_freq]');
xlabel('Time Steps');
ylabel('Frequency');
legend('Action 1', 'Action 2');
title('Evolution of Action Frequencies in State 1');




% Create a single figure with two subplots for 'Vs' and 'r_bar'
figure;
indx_ss2 = find(ss(1:end-1) == 2);
% Create the first subplot for 'Vs'
subplot(2, 1, 1);  % 2 rows, 1 column, first subplot
plot(indx_ss2, Vs(indx_ss2));
xlabel('Time steps');
ylabel('V(S=1)');


% Create the second subplot for 'r_bar'
subplot(2, 1, 2);  % 2 rows, 1 column, second subplot
plot(r_bar(1:end-1));
xlabel('Time steps');
ylabel('Mean reward');

% Adjust spacing between subplots
spacing = 0.04;
position = get(gcf, 'Position');
position(3) = position(3) + spacing;
set(gcf, 'Position', position);

% Define your data
rew_indx = find(as(1:end-1) == 1 & ss(1:end-1) == 1);
others_indx = find(as(1:end-1) == 2 & ss(1:end-1) == 1);
% Create a single figure with two subplots
figure;
% Create the first subplot for 'rew_indx'
bar(ls(rew_indx));
yticks(latencies(1:10:end));
xlabel('Reward action');
ylabel('Vigor');

% Adjust spacing between subplots
spacing = 0.04;
position = get(gcf, 'Position');
position(3) = position(3) + spacing;
set(gcf, 'Position', position);

% Create a single figure with all subplots
% figure;
% for action = 1:nA
%     for latency = 1:nL
%         subplot(nA, nL, (action - 1) * nL + latency);  % Create a subplot grid
%         plot(squeeze(Qs(1:end-1, action, latency)));
%         title(['A ', num2str(action), ', L ', num2str(latencies(latency))]);
%         xlabel('Time Step');
%         ylabel('Q-Value');
%         ylim([0 1]);  % Set y-axis limits to 0 and 1
%     end
% end

figure;
for latency = 1:nL
    scatter(latencies(latency), Qs(end-1, 1, latency));
    hold on
end
xlabel('Latency');
ylabel('Q*-Value');


figure()
plot(time_spent(1:end-1),storing(1:end-1))
xlabel('Real time');
ylabel('Rewards - Costs');

indx_ss1_r = find(ss(1:end-1) == 1 & as(1:end-1) == 1);
indx_ss1_nr = find(ss(1:end-1) == 1 & as(1:end-1) == 2);
indx_ss2 = find(ss(1:end-1) == 2);
figure()
plot(indx_ss1_r,delta(indx_ss1_r));
hold on
plot(indx_ss1_nr,delta(indx_ss1_nr));
hold on
plot(indx_ss2,delta(indx_ss2));
xlabel('Time steps');
ylabel('RPE');
legend('S1 reaward','S1 no reaward','S2')

figure()
plot(drug(1:end-1))
xlabel('Time steps');
ylabel('Drug in agent');



