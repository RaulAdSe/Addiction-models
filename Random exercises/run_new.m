clear; clc;
% Runs 1508
days = 10;
NT = 24000;
NTC = 6000;

nS = 4;
nA = 2;

Q_0 = zeros(nS,nA);       % pairs of state action values
r_bar_0 = 0;

as_all = [];
ss_all = [];
r_all = [];
r_bar_all = [];
delta_all = [];
Vs_all = zeros(NT*days,nS,nA);

for i = 1:days
        fprintf('day %d\n', i)
        [ss,as,r,delta,Vs,r_bar] = ex1508_fun(NTC,r_bar_0,Q_0);
        as_all = [as_all as(1:end-1)];
        ss_all = [ss_all ss(1:end-1)];
        r_all = [r_all r(1:end-1)];
        r_bar_all = [r_bar_all r_bar(1:end-1)];
        Vs_all(1 + (i-1)*NT:i*NT,:,:) = Vs(1:end-1,:,:);
        delta_all = [delta_all delta(1:end-1)];
        
        r_bar_0 = r_bar(end-1);
        Q_0(1,:) = Vs(end-1,1,:);
        Q_0(2,:) = Vs(end-1,2,:);
        
end

ss_c = ss_all(:);
as_c = as_all(:);
r_bar_c = r_bar_all(:);
delta_c = delta_all(:);
r_c = r_all(:);

figure;
subplot(3, 1, 1);
plot(find(ss_c(1:end-1) == 4), delta_c(ss_c(1:end-1) == 4));
hold on
plot(find(ss_c(1:end-1) ~= 4), delta_c(ss_c(1:end-1) ~= 4));
xlabel('Time steps');
ylabel('Delta');
legend('Bounded common state (4)','Others')
grid on;

subplot(3, 1, 2);
plot(r_bar_c);
xlabel('Time steps');
ylabel('Mean reward');
title('Mean reward');
grid on;

subplot(3, 1, 3);
actions_count = histcounts(as_c(ss_c == 1), 1:nA+1);
bar(actions_count / length(as_c(ss_c == 1)));
xlabel('Action');
ylabel('Frequency');
title('Frequency of Actions in State 1');
xticks(1:nA);
xticklabels({'Action 1', 'Action 2'}); % Update with actual action labels
grid on;

figure;
for s = 1:nS
    state_mask = ss_c(1:end-1) == s;
    
    subplot(nS, 3, 3*(s-1) + 1);
    plot(find(state_mask), delta_c(state_mask));
    xlabel('Time steps');
    ylabel('Delta');
    title(['Delta  - State ' num2str(s)]);
    grid on;
    
    subplot(nS, 3, 3*(s-1) + 2);
    plot(find(state_mask), r_bar_c(state_mask));
    xlabel('Time steps');
    ylabel('Mean reward');
    title(['Mean reward  - State ' num2str(s)]);
    grid on;
    
    subplot(nS, 3, 3*(s-1) + 3);
    actions_count = histcounts(as_c(state_mask), 1:nA+1);
    bar(actions_count / sum(state_mask));
    xlabel('Action');
    ylabel('Frequency');
    title(['Frequency of Actions in State 1 - State ' num2str(s)]);
    xticks(1:nA);
    xticklabels({'Action 1', 'Action 2'}); % Update with actual action labels
    grid on;
end

% Create a separate figure for plotting the temporal evolution of Q action-state values
figure;
for s = 1:nS
    subplot(nS, 1, s);
    Q_values = squeeze(Vs_all(1:end-1, s, :));
    plot(Q_values);
    xlabel('Time steps');
    ylabel('Q Value');
    title(['Q Action-State Values - State ' num2str(s)]);
    legend('Action 1', 'Action 2'); % Update with actual action labels
    grid on;
end

% Calculate the cumulative sum of ones in 'r'
cumulative_ones = cumsum(r_c);
% Calculate the temporal evolution of the frequency of ones
frequency_of_ones = cumulative_ones'./ (1:length(r_c));
% Create a figure and plot the frequency of ones
figure;
plot(1:length(r_c), frequency_of_ones, 'b');
xlabel('Time steps');
ylabel('Frequency of Ones');
title('Temporal Evolution of Frequency of Ones in Vector r');

% Set the chunk size
chunk_size = NT;
% Calculate the number of chunks
num_chunks = numel(r_c) / chunk_size;

% Initialize a vector to store the counts
chunk_counts = zeros(num_chunks, 1);

% Loop through the chunks
for chunk_idx = 1:num_chunks
    start_idx = (chunk_idx - 1) * chunk_size + 1;
    end_idx = chunk_idx * chunk_size;
    
    % Count the number of ones in the current chunk
    chunk_counts(chunk_idx) = sum(r_c(start_idx:end_idx));
end

% Plot the chunk counts
figure;
bar(chunk_counts);
xlabel('Day');
ylabel('Number of Ones');
title('Number of Reward each Day');




