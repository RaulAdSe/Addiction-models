% Same but with simplified Dezfouli approach
% 
clear ; clc;
N = 1;
nA = 2;
nS = (2)*N + 2;
T = zeros(nS, nS, nA);
T(:, :, 1) = [0 1 0 0; 0 0 0 1;0 0 0 1; 1 0 0 0]';
T(:, :, 2) = [0 0 1 0; 0 0 0 1;0 0 0 1; 1 0 0 0]';
R = zeros(nS, nA);
R(1,1) = 1;


NT = 10000;
NTC = 10000;

ss = zeros(NT + 1, 1);
ss(1) = 1;
r = zeros(NT, 1);
r_bar = zeros(NT + 1, 1);
as = zeros(NT + 1, 1);
delta = zeros(NT + 1, 1);
Q = zeros(nS,nA);
Vs = zeros(NT + 1, nS,nA);

Ds = 15;
beta = 5e-2; 
alpha = 1;
epsi = 5e-2;

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
    
    if t == NTC
        R(1,1) = 0;
    end
    
    ss(t + 1) = find(T(:, ss(t), as(t)));
    r(t) = R(ss(t), as(t));

    V = max(Q(ss(t+1),:));

    if ss(t) == 1 && as(t) == 1 && t < NTC
        delta(t)= max(r(t) + V - Q(ss(t), as(t)) + Ds, Ds) - r_bar(t);
    else
        delta(t) = r(t) + V - Q(ss(t), as(t)) - r_bar(t);
    end
    
    r_bar(t + 1) = r_bar(t) + beta * delta(t);

    if ss(t) == 4   % Fixed value state
        Q(ss(t),:) = 0;
    else
        Q(ss(t),as(t)) = Q(ss(t),as(t)) + alpha * delta(t);
    end

    Vs(t,:,:) = Q(:,:); % fix one so the others converge
end

figure;
subplot(3, 1, 1);
plot(find(ss(1:end-1) == 4), delta(ss(1:end-1) == 4));
hold on
plot(find(ss(1:end-1) == 1 & as(1:end-1) == 1), delta(ss(1:end-1) == 1 & as(1:end-1) == 1));
hold on
plot(find(ss(1:end-1) ~= 4), delta(ss(1:end-1) ~= 4));
xlabel('Time steps');
ylabel('Delta');
legend('Bounded common state (4)','Drug','Others')
grid on;

subplot(3, 1, 2);
plot(r_bar);
xlabel('Time steps');
ylabel('Mean reward');
title('Mean reward');
grid on;

subplot(3, 1, 3);
actions_count = histcounts(as(ss == 1), 1:nA+1);
bar(actions_count / length(as(ss == 1)));
xlabel('Action');
ylabel('Frequency');
title('Frequency of Actions in State 1');
xticks(1:nA);
xticklabels({'Action 1', 'Action 2'}); % Update with actual action labels
grid on;

% Adjust spacing between subplots
subplot_adjust = get(gcf, 'Position');
subplot_adjust(4) = subplot_adjust(4) - 0.05;
set(gcf, 'Position', subplot_adjust)

figure;
for s = 1:nS
    state_mask = ss(1:end-1) == s;
    
    subplot(nS, 3, 3*(s-1) + 1);
    plot(find(state_mask), delta(state_mask));
    xlabel('Time steps');
    ylabel('Delta');
    title(['Delta  - State ' num2str(s)]);
    grid on;
    
    subplot(nS, 3, 3*(s-1) + 2);
    plot(find(state_mask), r_bar(state_mask));
    xlabel('Time steps');
    ylabel('Mean reward');
    title(['Mean reward  - State ' num2str(s)]);
    grid on;
    
    subplot(nS, 3, 3*(s-1) + 3);
    actions_count = histcounts(as(state_mask), 1:nA+1);
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
    Q_values = squeeze(Vs(1:end-1, s, :));
    plot(Q_values);
    xlabel('Time steps');
    ylabel('Q Value');
    title(['Q Action-State Values - State ' num2str(s)]);
    legend('Action 1', 'Action 2'); % Update with actual action labels
    grid on;
end

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
one_indices = find(r == 1);

% Calculate the time steps between consecutive ones
time_steps_between_ones = diff(one_indices);

% Create a figure and plot the time steps between consecutive ones
figure;
plot(1:length(time_steps_between_ones), time_steps_between_ones, 'b');
xlabel('Consecutive Occurrences of Ones');
ylabel('Time Steps Between Ones');
title('Inter Reward Interval');
grid on;