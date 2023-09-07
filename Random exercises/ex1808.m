% Do the same as yesterday but with a MDP that could induce negative
% deltas when the drug is removed. Implement the Keramati MDP with 2 actions.
clear;clc;close all;

NT = 2000;
N_C = 1000;

nS = 5; % number of states, each occupied for 4 seconds = 20s
nA = 2; % number of actions: 1=L (lever), 2=OTH (other)

T = zeros(nS,nS,nA);
T(:,:,2) = [1 0 0 0 0; 0 0 1 0 0; 0 0 0 1 0; 0 0 0 0 1; 1 0 0 0 0]'; % col=s(t), row=s(t+1)
T(:,:,1) = T(:,:,2);
T(:,1,1) = [0 1 0 0 0]'; % the only difference for ALP, is that choosing it in state 1 leads to transition to 2
R = zeros(nS,nA);
R(1,1) = 1;

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
alpha_minus = 1e-2;

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
    
    if t == N_C
        R(1,1) = 0;
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

    if ss(t) == nS   % Fixed value state
        Q(ss(t),:) = 0;
    else
        Q(ss(t),as(t)) = Q(ss(t),as(t)) + alpha * delta(t);
    end

    Vs(t,:,:) = Q(:,:); % fix one so the others converge
end

figure;
subplot(3, 1, 1);
plot(find(ss(1:end-1) == nS), delta(ss(1:end-1) == nS));
hold on
plot(find(ss(1:end-1) == 1 & as(1:end-1) == 1), delta(ss(1:end-1) == 1 & as(1:end-1) == 1));% hold on
% plot(setdiff([1:NT-1], union(ss(1:end-1) == 4,ss(1:end-1) == 1 & as(1:end-1) == 1)), delta(setdiff([1:NT-1], union(ss(1:end-1) == 4,ss(1:end-1) == 1 & as(1:end-1) == 1))));
xlabel('Time steps');
ylabel('Delta');
legend('Bounded common state (N)','Drug intakes')
grid on;

subplot(3, 1, 2);
plot([1:NT],r_bar(1:end-1));
xlabel('Time steps');
ylabel('Mean reward');
title('Mean reward');
grid on;

% Set the chunk size
chunk_size = 1;
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

figure;

% Define the number of rows and columns for the subplot grid
num_rows = 4; % Define the number of rows
num_cols = ceil(nS / num_rows); % Calculate the number of columns

for s = 1:nS
    subplot(num_rows, num_cols, s); % Use the subplot grid layout
    Q_values = squeeze(Vs(1:end-1, s, :));
    plot([1:NT],Q_values);
    xlabel('Time steps');
    ylabel('Q Value');
    title(['State ' num2str(s)]); % Add a title with the state number
    grid on;
end
legend('Action 1', 'Action 2'); % Update with actual action labels


% Adjust spacing between subplots
subplot_adjust = get(gcf, 'Position');
subplot_adjust(4) = subplot_adjust(4) - 005;
set(gcf, 'Position', subplot_adjust)

