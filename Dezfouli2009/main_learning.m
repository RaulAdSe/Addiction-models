% Dezfouli's model

clear; clc; close all
% t_step = 40;
% 
% lss = 1; % number of hours access for short-access condition
% lsl = 6; % number of hours access for long-access condition
% 
% for ii = 1:2
%     
%     if ii ==1
%         condition = 'short'; % 'short' or 'long' (refers to length of access to cocaine)
%     else
%         condition = 'long';
%     end
%     
%     if strcmp(condition,'short')
%        length_sess = lss;
%     elseif strcmp(condition,'long')
%         length_sess = lsl;
%     else
%         error('incorrect specification of condition!')
%     end
%     days = 25;

NT = 2000;

    % initial conditions (i.e., for first day)
    % we assume that at the beggining of each training h has dropped to 0,
    % which is not crazy since it follows an exponential decay
 
    % EL PRIMER COP Q ES TOTA 0s!! DESPRËS NO!
    
    nS = 5; % number of 'external' states
    nA = 3; % number of actions

% sometimes the T and R matrixes are called the action-state graph
    Q = zeros(nS,nA);       % pairs of state action values

%     as_all = [];
%     ss_all = [];
%     rs_all = [];
%     rc_all = [];
%     delta_all = [];
%     r_bar_all = [];
%     kappa_all = [];
%     Vs_all = [];
    r_bar_o = 0;
    kappa_o = 0;
%     for i = 1:days
%         fprintf('day %d\n', i)
        [as, ss, r, r_e, r_bar, kappa, delta, Vs, Q, plt] = simulate_a_day_l(NT, Q, kappa_o, r_bar_o);               % Change here the RL algorithm
%         as_all = [as_all as];
%         ss_all = [ss_all ss];
%         rs_all = [rs_all r];
%         rc_all = [rc_all r_c];
%         r_bar_all = [r_bar_all r_bar];
%         delta_all = [delta_all delta];
%         kappa_all = [kappa_all kappa];
%         Vs_all = [Vs_all Vs];
%         
%         r_bar_o = r_bar(end);
%         kappa_o = kappa(end);
%         
%     end

    % plot results
    figure(1)
    hold on
    num_panels = 10;
    num_rows = 2;
    %
    subplot(2,num_panels/num_rows,1)
    hold on
    for a = 1:nA
        plot([1:NT], Vs(1:end-1,1,a));
        hold on
    end
    xlabel('Time steps');
    ylabel('Action value in state 1');
    legend('OTH', 'IL', 'AL');

    %
    subplot(2,num_panels/num_rows,2)
    hold on
    % Initialize variables for plotting
    num_elements = NT;
    count = [];

    % Loop through the data in chunks of 100 elements
    for i = 1:100:num_elements
        chunk_end = min(i + 99, num_elements);
        chunk_as = as(i:chunk_end);
        chunk_ss = ss(i:chunk_end);

        % Count elements that satisfy the condition
        condition_count = sum(chunk_as == 3 & chunk_ss == 1);

        % Store count and x value
        count = [count, condition_count];
    end

    % Plot the count
    plot(count, 'o-');
    xlabel('Chunk Number');
    ylabel('Drug intake events');
    %
    subplot(2,num_panels/num_rows,3)
    hold on
    plot([1:NT],kappa(1:end-1));
    xlabel('Time steps'), ylabel('Kappa')
    %
    
    drugindx = find(ss(:) == 1 & as(:) == 3);
    punishindx = drugindx + 1;
    otherindx = setdiff(1:NT, union(drugindx, punishindx));
    
    subplot(2,num_panels/num_rows,4)
    hold on
%     plot(drugindx,r_bar(drugindx));
%     hold on
%     plot(punishindx,r_bar(punishindx));
% 	hold on
%     plot(otherindx,r_bar(otherindx));
    plot(r_bar(:));
    xlabel('Time steps'), ylabel('Mean reward')
    %
    
    subplot(2,num_panels/num_rows,5)
    % Generate data for testing (you can replace this with your actual data)
    num_elements = NT;
    % Initialize variables for calculations and plotting
    chunk_size = 100;
    num_chunks = ceil(num_elements / chunk_size);
    mean_time_diffs = zeros(num_chunks, 1);

    % Loop through the data in chunks
    for i = 1:num_chunks
        chunk_start = (i - 1) * chunk_size + 1;
        chunk_end = min(i * chunk_size, num_elements);

        chunk_as = as(chunk_start:chunk_end);
        chunk_ss = ss(chunk_start:chunk_end);

        % Find indices of events satisfying the condition
        event_indices = find(chunk_as == 3 & chunk_ss == 1);

        % Calculate time differences between consecutive events
        event_diffs = diff(event_indices);

        % Calculate the mean time difference for the chunk
        mean_time_diffs(i) = mean(event_diffs);
    end

    % Plot the mean time differences for each chunk
    plot(mean_time_diffs,'o-');
    xlabel('Chunk Number');
    ylabel('Mean IDI');
    %
    
    subplot(2,num_panels/num_rows,6)
    hold on
    plot(drugindx, r_e(drugindx));
    hold on
    plot(punishindx,r_e(punishindx));
    hold on
    plot(otherindx,r_e(otherindx));

    xlabel('Time steps')
    ylabel('Experienced reward')
    legend('Drug','Punishment','No drug')
    %
%     subplot(2,num_panels/num_rows,7)
%     hold on
%     plot([1:NT],delta_c(1:end-1));
%     xlabel('Time steps'), ylabel('RPE under cocaine')
    
    subplot(2,num_panels/num_rows,7)
    hold on
    plot(drugindx, delta(drugindx));
    ylabel('RPE drug')
    xlabel('Time steps')
    %
    subplot(2,num_panels/num_rows,8)
    hold on
    plot(otherindx,delta(otherindx));
    ylabel('RPE no drug')
    xlabel('Time steps')


    subplot(2,num_panels/num_rows,9)    % Action dynamics
    hold on

    % Set the chunk size and number of chunks
    chunk_size = 100;
    num_chunks = floor(numel(as) / chunk_size);

    % Initialize action counts for each chunk
    action_counts = zeros(num_chunks, 3);

    % Count actions in each chunk
    for chunk = 1:num_chunks
        chunk_indices = (chunk - 1) * chunk_size + 1 : chunk * chunk_size;
        chunk_as = as(chunk_indices);
        action_counts(chunk, 1) = sum(chunk_as == 1);
        action_counts(chunk, 2) = sum(chunk_as == 2);
        action_counts(chunk, 3) = sum(chunk_as == 3);
    end

    % Plot the action counts for each chunk
    plot(action_counts);
    xlabel('Chunk Number');
    ylabel('Action Counts');
    legend('OTH', 'IL', 'AL');
    
    subplot(2,num_panels/num_rows,10)    % State dynamics
    hold on
    % Set the chunk size and number of chunks
    chunk_size = 100;
    num_chunks = floor(numel(ss) / chunk_size);

    % Initialize state visit counts for each chunk
    state_counts = zeros(num_chunks, 5);

    % Count state visits in each chunk
    for chunk = 1:num_chunks
        chunk_indices = (chunk - 1) * chunk_size + 1 : chunk * chunk_size;
        chunk_ss = ss(chunk_indices);
        for state = 1:5
            state_counts(chunk, state) = sum(chunk_ss == state);
        end
    end

    % Plot the state visit counts for each chunk
    plot(state_counts);
    xlabel('Chunk Number');
    ylabel('State Visit Counts');
    legend('State 1', 'State 2', 'State 3', 'State 4', 'State 5');
   
figure();

for value = 1:nS
    subplot(2, 3, value); % Adjust subplot dimensions as needed
    chunk_size = 100;
    num_chunks = floor(numel(ss) / chunk_size);
    actionsCountPerDay = zeros(num_chunks, nA);

    for chunk = 1:num_chunks
        chunk_indices = (chunk - 1) * chunk_size + 1 : chunk * chunk_size;
        chunk_ss = ss(chunk_indices);
        chunk_as = as(chunk_indices);
        stateValueIndicesDay = find(chunk_ss(:) == value);
        for action = 1:nA
            actionsCountPerDay(chunk, action) = sum(chunk_as(stateValueIndicesDay) == action);
        end
    end

    plot(actionsCountPerDay, 'o-');
    xlabel('Chunk Number');
    ylabel('Action Counts');
    title(['State ', num2str(value)]);
end
    legend('OTH', 'IL', 'AL');

figure;

% Create a subplot for the first plot
subplot(2,1,1);
plot(punishindx, delta(punishindx));
ylabel('RPE punishment')
xlabel('Time steps')
% Create a subplot for the second plot
subplot(2,1,2);
state5_indx = find(ss(:) == 5);
state1_indx = find(ss(:) == 1);
other_indx = setdiff(1:NT, union(state5_indx, state1_indx));
hold on
plot(state5_indx, r_bar(state5_indx));
hold on
plot(state1_indx, r_bar(state1_indx));
hold on
plot(other_indx, r_bar(other_indx));
legend('State 5', 'State 1', 'Other')
xlabel('Time steps'), ylabel('Mean reward')

figure();
for s = 1:nS
    % Create a subplot for the current state
    subplot(5, 1, s);
    % Plot the evolution of each action's value across timesteps
    for a = 1:nA
        plot(Vs(:, s, a));
        hold on;
    end
    
    title(['State ' num2str(s)]);
    xlabel('Time steps');
    ylabel('Value');
    grid on;
end
    legend('OTH', 'IL', 'AL');

figure()
for ind = 1:nS
    state_indx = find(ss(1:end-1) == ind);
    plot(state_indx,plt(state_indx))
    hold on
end

xlabel('Time steps');
ylabel('V(s_{t+1})');
legend('State 1','State 2','State 3','State 4','State 5')

% figure()
% state_indx = find(ss(:) == 1 & as(:) == 3);
% plot(state_indx,plt(state_indx))
% hold on
% state_indx = find(ss(:) == 1 & as(:) == 1);
% plot(state_indx,plt(state_indx))
% state_indx = find(ss(:) == 1 & as(:) == 2);
% plot(state_indx,plt(state_indx))
% xlabel('Time steps');
% ylabel('Q-V');
% legend('Cocaine','OTH','IL')
% 
% figure()
% state_1_nodrug = find(ss(:) == 1 & as(:) ~= 3);
% state_others = find(ss(:) ~= 1);
% plot(state_1_nodrug,delta(state_1_nodrug));
% hold on
% plot(state_others,delta(state_others));
% xlabel('Time steps');
% ylabel('RPE no drug');
% legend('State 1','Other states')