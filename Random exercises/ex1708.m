% clear ; clc; close all;
% Describes a training procedure over a 24 state MDP. Drug only available
% over the first N_C ones.
NT = 24*10000;
N = 24;
N_C = 6;
nA = 2;
nS = N;
T = zeros(nS, nS, nA);

for i = 1:N
    for j = 1:N-1
        if i == j
            T(i+1,j,:) = 1;
        end
    end
end
T(1,end,:) = 1;

R = zeros(nS, nA);
for i = 1: N_C
    R(i,1) = 1;
end

ss = zeros(NT + 1, 1);
ss(1) = 1;
r = zeros(NT, 1);
r_bar = zeros(NT + 1, 1);
as = zeros(NT + 1, 1);
delta = zeros(NT + 1, 1);
Q = zeros(nS,nA);
Vs = zeros(NT + 1, nS,nA);

beta_plus = 1e-2; 
beta_minus = 1e-3; 
alpha_plus = 1e-1;
alpha_minus = 1e-1;

epsi = 1e-1;

for t = 1:NT

    exploratory = false;
    if rand() < epsi
        exploratory = true;
        as(t) = randi(numel(Q(ss(t), :)));
    else
        value_dict = zeros(1, nA);  % store estimated values of diferent actions in current sate
        for a = 1:nA
            value_dict(a) = Q(ss(t),a);
        end
        [~, max_idx] = max(value_dict);
        as(t) = max_idx;
    end
    
    ss(t + 1) = find(T(:, ss(t), as(t)));
    r(t) = R(ss(t), as(t));

    V = max(Q(ss(t+1),:));

    delta(t) = r(t) + V - Q(ss(t), as(t)) - r_bar(t);
       
    if delta(t) > 0
        beta = beta_plus;
        alpha = alpha_plus;
    else
        beta = beta_minus;
        alpha = alpha_minus;
    end

    r_bar(t + 1) = r_bar(t) + beta * delta(t);

    if ss(t) == N   % Fixed value state
        Q(ss(t),:) = 0;
    else
        Q(ss(t),as(t)) = Q(ss(t),as(t)) + alpha * delta(t);
    end

    Vs(t,:,:) = Q(:,:); % fix one so the others converge
end

figure;
subplot(3, 1, 1);
plot(find(ss(1:end-1) == N)/N, delta(ss(1:end-1) == N));
hold on
plot(find(ismember(ss(1:end-1), [1:N_C]) & as(1:end-1) == 1)/N, delta(ismember(ss(1:end-1), [1:N_C]) & as(1:end-1) == 1));
% hold on
% plot(setdiff([1:NT-1], union(ss(1:end-1) == 4,ss(1:end-1) == 1 & as(1:end-1) == 1))/N, delta(setdiff([1:NT-1], union(ss(1:end-1) == 4,ss(1:end-1) == 1 & as(1:end-1) == 1))));
xlabel('Days');
ylabel('Delta');
legend('Bounded common state (N)','Drug intakes')
grid on;

subplot(3, 1, 2);
plot([1:NT]./N,r_bar(1:end-1));
xlabel('Days');
ylabel('Mean reward');
title('Mean reward');
grid on;

% Set the chunk size
chunk_size = N*5;
% Calculate the number of chunks
num_chunks = numel(r) / chunk_size;
% Initialize a vector to store the counts
chunk_counts = zeros(num_chunks, 1);

% Loop through the chunks
for chunk_idx = 1:num_chunks
    start_idx = (chunk_idx - 1) * chunk_size + 1;
    end_idx = chunk_idx * chunk_size;
    % Count the number of ones in the current chunk
    chunk_counts(chunk_idx) = sum(r(start_idx:end_idx));
end
subplot(3, 1, 3);
bar(chunk_counts);
ylabel('Number of Ones');
title('Number of Rewards each Day');



% plot(r - r_bar(1:end-1));
% xlabel('Time steps');
% ylabel('r-mean reward');
% title('Quantity');
% grid on;

% % Adjust spacing between subplots
% subplot_adjust = get(gcf, 'Position');
% subplot_adjust(4) = subplot_adjust(4) - 0.05;
% set(gcf, 'Position', subplot_adjust)
% 
% figure;
% for s = 1:nS
%     state_mask = ss(1:end-1) == s;
%     
%     subplot(nS, 3, 3*(s-1) + 1);
%     plot(find(state_mask), delta(state_mask));
%     xlabel('Time steps');
%     ylabel('Delta');
%     title(['Delta  - State ' num2str(s)]);
%     grid on;
%     
%     subplot(nS, 3, 3*(s-1) + 2);
%     plot(find(state_mask), r_bar(state_mask));
%     xlabel('Time steps');
%     ylabel('Mean reward');
%     title(['Mean reward  - State ' num2str(s)]);
%     grid on;
%     
%     subplot(nS, 3, 3*(s-1) + 3);
%     actions_count = histcounts(as(state_mask), 1:nA+1);
%     bar(actions_count / sum(state_mask));
%     xlabel('Action');
%     ylabel('Frequency');
%     title(['Frequency of Actions in State 1 - State ' num2str(s)]);
%     xticks(1:nA);
%     xticklabels({'Action 1', 'Action 2'}); % Update with actual action labels
%     grid on;
% end
% 
% % Create a separate figure for plotting the temporal evolution of Q action-state values
figure;

% Define the number of rows and columns for the subplot grid
num_rows = 4; % Define the number of rows
num_cols = ceil(nS / num_rows); % Calculate the number of columns

for s = 1:nS
    subplot(num_rows, num_cols, s); % Use the subplot grid layout
    Q_values = squeeze(Vs(1:end-1, s, :));
    plot([1:NT]./N,Q_values);
    xlabel('Days');
    ylabel('Q Value');
    title(['State ' num2str(s)]); % Add a title with the state number
    grid on;
end
legend('Action 1', 'Action 2'); % Update with actual action labels


% Adjust spacing between subplots
subplot_adjust = get(gcf, 'Position');
subplot_adjust(4) = subplot_adjust(4) - 0.05;
set(gcf, 'Position', subplot_adjust)

% % Calculate the cumulative sum of ones in 'r'
% cumulative_ones = cumsum(r);
% % Calculate the temporal evolution of the frequency of ones
% frequency_of_ones = cumulative_ones'./ (1:length(r));
% % Create a figure and plot the frequency of ones
% figure;
% plot(1:length(r), frequency_of_ones, 'b');
% xlabel('Time steps');
% ylabel('Frequency of Ones');
% title('Temporal Evolution of Frequency of Ones in Vector r');
% grid on;
% Find the indices of ones in the vector 'r'
% one_indices = find(r == 1);
% 
% % Calculate the time steps between consecutive ones
% time_steps_between_ones = diff(one_indices);
% 
% % Create a figure and plot the time steps between consecutive ones
% figure;
% plot(1:length(time_steps_between_ones), time_steps_between_ones, 'b');
% xlabel('Consecutive Occurrences of Ones');
% ylabel('Time Steps Between Ones');
% title('Inter Reward Interval');
% grid on;




