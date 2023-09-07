% let essentially replicate Mehdi's model.
% now with learning of transitions and outcomes (except costs of lever
% pressing).

clear; clc; close all

rng(0)
lss = 1; % number of hours access for short-access condition
lsl = 6; % number of hours access for long-access condition

for ii = 1:2
    
    if ii ==1
        condition = 'short'; % 'short' or 'long' (refers to length of access to cocaine)
    else
        condition = 'long';
    end
    
    
    if strcmp(condition,'short')
       length_sess = lss;
    elseif strcmp(condition,'long')
        length_sess = lsl;
    else
        error('incorrect specification of condition!')
    end
    days = 25;

    % initial conditions (i.e., for first day)
    XX_0 = 100; % threshold/setpoint, h*
    % we assume that at the beggining of each training h has dropped to 0,
    % which is not crazy since it follows an exponential decay
    X_0 = 0;    % internal state, h
    c_0 = 0;    % belief state
    nS = 5; % number of 'external' states
    nA = 3; % number of actions
    T_hat_nc = zeros(nS,nS,nA); % transition function, no-cocaine
    O_hat_nc = zeros(nS,nA); % outcome function, no-cocaine (excepting costs of actions)
    T_hat_c = zeros(nS,nS,nA); % transition function, cocaine
    O_hat_c = zeros(nS,nA); % outcome function, cocaine (excepting costs of actions)

    XXs_all = [];
    Xs_all = [];
    as_all = [];
    ss_all = [];
    Vs_all = [];
    bel_all = []; % agent's degree of belief of being in cocaine-available vs. cocaine-unavailable ??? IS THIS NOT THE EFFECT?
    for i = 1:days
        fprintf('day %d\n', i)
        [XXs, Xs, bbel, as, ss, T_hat_nc, O_hat_nc, T_hat_c, O_hat_c, Vs] = ...
            simulate_a_day_l( length_sess, XX_0, X_0, c_0, T_hat_nc, O_hat_nc, T_hat_c, O_hat_c );
        XXs_all = [XXs_all XXs];
        Xs_all = [Xs_all Xs];
        bel_all = [bel_all bbel];
        as_all = [as_all as];
        ss_all = [ss_all ss];
        Vs_all = [Vs_all Vs];        % aqui
        XX_0 = XXs(end);
    %     X_0 = Xs(end);
    %     c_0 = bbel(end);
%     O_hat_nc
%     O_hat_c
    end

    % summary measures?
%     Incorrect measures
%     ips = sum(as_all==3,1); % number of infusions per session/day 
%     iph = sum(as_all(1:(3600/4),:)==3); % number of infusions in first hour of session/day
    % Corrections made
    iptf = sum( reshape( (as_all(1:(3600/4),1)==3  & ss_all(1:(3600/4),1)==1), [600/4 6] ) );  % infusions per 10 minutes for first hour of first session
    iptl = sum( reshape( (as_all(1:(3600/4),end)==3 & ss_all(1:(3600/4),end)==1), [600/4 6] ) );% infusions per 10 minutes for first hour of last session
    idx_inf_ls = find(as_all(:,end)==3 & ss_all(:,end) == 1);
    iii = diff(idx_inf_ls(1:min(30,length(idx_inf_ls))))*4; % inter-infusion intervals in last session (first 30 infusions); recall each time step is 4 secs

    % plot results
    figure(1)
    hold on
    % suptitle([condition ' access (', num2str(length_sess), ' hr)'])
    num_panels = 8;
    num_rows = 2;
    %
    subplot(2,num_panels/num_rows,1)
    hold on
%     plot(ips,'o')
%     xlabel('session'), ylabel('infusions per session')

    plot([1:25],sum(as_all(1:1*60*60*length_sess/4,:) == 3 & ss_all(1:1*60*60*length_sess/4,:) == 1), 'o');
    xticks(1:4:25); % Set x-axis ticks to appear every 5 units
    xlabel('Day');
    ylabel('Infussions / Session');
    xlim([1, 25]); 
    %
    subplot(2,num_panels/num_rows,2)
    hold on
%     plot(iph,'o')
    plot([1:25],sum(as_all(1:1*60*60*1/4,:) == 3 & ss_all(1:1*60*60*1/4,:) == 1), 'o');
    xlabel('Session'), ylabel('Infusions / First Hour')
    xlim([1, 25]); 
    %
    subplot(2,num_panels/num_rows,3)
    hold on
    plot(XXs_all(:));
    xticks(linspace(1,length(XXs_all(:)),7)); 
    xticklabels(1:4:25); % Custom labels
    xlim([1,length(XXs_all(:))]);   % real xaxis lims
    ylim([80,220]);
    xlabel('Day'), ylabel('Homeostatic setpoint')
    yline(100, 'k--');
    yline(200, 'k--');
    % set(gca,'ylim',[95 125],'ytick',100:5:120)
    %
    subplot(2,num_panels/num_rows,4)
    if ii == 1
        plot(10:10:60,iptf,'bo-')
        hold on
        plot(10:10:60,iptl,'ro-')
    else
        plot(10:10:60, iptf, 'b^--');        
        hold on
        plot(10:10:60,iptl,'r^--')
    end
    set(gca,'ylim',[0 15], 'xtick',0:20:60)
    xlabel('Time (min)'), ylabel('Infusions / 10 min')
    legend({'First session','Last session'})
    %
    subplot(2,num_panels/num_rows,5)
    hold on
    plot((as_all(1:(30*60)/4,end) == 3 & ss_all(1:(30*60)/4,end) == 1))
    set(gca,'xtick',0:(10*60)/4:(30*60)/4,'ytick',[],'xticklabel',0:10:30, 'ylim', [0 1.5])
    xlabel('Time (min)'), ylabel('Infusions (last session)')
    %
    subplot(2,num_panels/num_rows,6)
    hold on

    % iii should be the length of max(Infussions/session). Becomes a
    % problem in short acess
    if ii == 1
        iii = iii(1:max(sum(as_all(1:1*60*60*length_sess/4,:) == 3 & ss_all(1:1*60*60*length_sess/4,:) == 1)));
    end
    plot(iii,'--o')
    
    if ii == 2  % Thus only set the xtick for the LgA since it will aready include the ShA
        set(gca,'xtick',0:5:30)
    end
    
    xlabel('Infusion number'), ylabel('Inter-Infusion intervals (sec)')
    subplot(2,num_panels/num_rows,7)
    hold on
    plot(Xs(1:(30*60)/4,end))
    set(gca,'xtick',0:(10*60)/4:(30*60)/4,'xticklabel',0:10:30)
    xlabel('Time (min)'), ylabel('Internal State (last session)')
    hold on
    
    % Storing for future comparisons
    if ii == 1
        as_all_s = as_all;
        ss_all_s = ss_all;
    end
    
    subplot(2,num_panels/num_rows,8)
    hold on
    plot(Vs_all(:))
%     set(gca,'xtick',0:(10*60)/4:(30*60)/4,'ytick',[],'xticklabel',0:10:30, 'ylim', [0 1.5])
    xlabel('Time steps'), ylabel('Drug value')
    
end

% Define legend labels and colors
legend_labels = {'ShA', 'LgA'};
legend_colors = {[0, 0.4470, 0.7410], [1, 0.5, 0]}; % Blue and Orange
legend_handle = legend(legend_labels, 'Location', 'best'); % Create the legend

% Set the legend colors to match the actual plot colors
set(legend_handle, 'TextColor', 'k'); % Set legend text color to black
legend_colorboxes = findobj(legend_handle, 'type', 'patch');

% Plot about strategies regarding actions and values
figure()
num_panels = 6;

subplot(2,num_panels/num_rows,1)    % Action dynamics
hold on

% Set the chunk size and number of chunks

% Initialize action counts for each chunk
action_counts = zeros(days, 3);

% Count actions in each chunk
for chunk = 1:days
    chunk_as = as_all_s(:,chunk);
    action_counts(chunk, 1) = sum(chunk_as == 1);
    action_counts(chunk, 2) = sum(chunk_as == 2);
    action_counts(chunk, 3) = sum(chunk_as == 3);
end

% Plot the action counts for each chunk
plot(action_counts);
xlabel('Days');
ylabel('Action Counts');
legend('OTH', 'IL', 'AL');

subplot(2,num_panels/num_rows,2)    % State dynamics
hold on
% Set the chunk size and number of chunks

% Initialize state visit counts for each chunk
state_counts = zeros(days, 5);

% Count state visits in each chunk
for chunk = 1:days
    chunk_ss = ss_all_s(:,chunk);
    for state = 1:5
        state_counts(chunk, state) = sum(chunk_ss == state);
    end
end

% Plot the state visit counts for each chunk
plot(state_counts);
xlabel('Days');
ylabel('State Visit Counts');
legend('State 1', 'State 2', 'State 3', 'State 4', 'State 5');


% actions taken in state 1 only evolving in time
subplot(2,num_panels/num_rows,3)    % Action dynamics

% Initialize variables
actionsCountPerDay = zeros(days, nA);

% Count actions taken in state 1 for each day
for day = 1:days
    stateOneIndicesDay = find(ss_all_s(:, day) == 1);
    for action = 1:nA
        actionsCountPerDay(day, action) = sum(as_all_s(stateOneIndicesDay, day) == action);
    end
end

plot(1:days, actionsCountPerDay, 'o-');
xlabel('Day');
ylabel('Action Count');
legend('OTH', 'IL', 'AL');



subplot(2,num_panels/num_rows,4)    % Action dynamics
hold on

% Set the chunk size and number of chunks

% Initialize action counts for each chunk
action_counts = zeros(days, 3);

% Count actions in each chunk
for chunk = 1:days
    chunk_as = as_all(:,chunk);
    action_counts(chunk, 1) = sum(chunk_as == 1);
    action_counts(chunk, 2) = sum(chunk_as == 2);
    action_counts(chunk, 3) = sum(chunk_as == 3);
end

% Plot the action counts for each chunk
plot(action_counts);
xlabel('Days');
ylabel('Action Counts');
legend('OTH', 'IL', 'AL');

subplot(2,num_panels/num_rows,5)    % State dynamics
hold on
% Set the chunk size and number of chunks

% Initialize state visit counts for each chunk
state_counts = zeros(days, 5);

% Count state visits in each chunk
for chunk = 1:days
    chunk_ss = ss_all(:,chunk);
    for state = 1:5
        state_counts(chunk, state) = sum(chunk_ss == state);
    end
end

% Plot the state visit counts for each chunk
plot(state_counts);
xlabel('Days');
ylabel('State Visit Counts');
legend('State 1', 'State 2', 'State 3', 'State 4', 'State 5');

% actions taken in state 1 only evolving in time
subplot(2,num_panels/num_rows,6)    % Action dynamics

% Initialize variables
actionsCountPerDay = zeros(days, nA);

% Count actions taken in state 1 for each day
for day = 1:days
    stateOneIndicesDay = find(ss_all(:, day) == 1);
    for action = 1:nA
        actionsCountPerDay(day, action) = sum(as_all(stateOneIndicesDay, day) == action);
    end
end

plot(1:days, actionsCountPerDay, 'o-');
xlabel('Day');
ylabel('Action Count');
legend('OTH', 'IL', 'AL');
