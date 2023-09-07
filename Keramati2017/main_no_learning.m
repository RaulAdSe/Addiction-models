% let essentially replicate Mehdi's model

clear; clc; %close all

rng(0)
lss = 1; % number of hours access for short-access condition
lsl = 6; % number of hours access for long-access condition
condition = 'long'; % 'short' or 'long' (refers to length of access to cocaine)
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
X_0 = 0;    % internal state, h

XXs_all = [];
Xs_all = [];
as_all = [];
ss_all = [];
for i = 1:days
    fprintf('day %d\n', i)
    [XXs, Xs, as, ss] = simulate_a_day_nl( length_sess, XX_0, X_0 );
    XXs_all = [XXs_all XXs];
    Xs_all = [Xs_all Xs];
    as_all = [as_all as];
    ss_all = [ss_all ss];
    XX_0 = XXs(end);
    X_0 = Xs(end);
end

% summary measures?
ips = sum(as_all==3); % number of infusions per session/day 
iph = sum(as_all(1:(3600/4),:)==3); % number of infusions in first hour of session/day
iptf = sum( reshape( (as_all(1:(3600/4),1)==3), [600/4 6] ) );  % infusions per 10 minutes for first hour of first session
iptl = sum( reshape( (as_all(1:(3600/4),end)==3), [600/4 6] ) );% infusions per 10 minutes for first hour of last session
idx_inf_ls = find( as_all(:,end)==3 );
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
plot(ips,'o')
xlabel('session'), ylabel('infusions per session')
%
subplot(2,num_panels/num_rows,2)
hold on
plot(iph,'o')
xlabel('session'), ylabel('infusions per session (first hour)')
%
subplot(2,num_panels/num_rows,3)
hold on
plot(XXs_all(:)); set(gca,'xtick',size(XXs_all,1):size(XXs_all,1):length(XXs_all(:)), 'xticklabels', 1:1:days)
xlabel('session/day'), ylabel('setpoint')
% set(gca,'ylim',[95 125],'ytick',100:5:120)
%
subplot(2,num_panels/num_rows,4)
plot(10:10:60,iptf,'bo-')
hold on
plot(10:10:60,iptl,'r^--')
set(gca,'ylim',[0 10], 'xtick',10:10:60)
xlabel('time (mins)'), ylabel('infusions per 10 mins')
legend({'first session','last session'})
%
subplot(2,num_panels/num_rows,5)
hold on
plot(ss(1:(30*60)/4,end),'k')
set(gca,'xtick',0:(10*60)/4:(30*60)/4,'ytick',[],'xticklabel',0:10:30, 'ylim', [1 6])
xlabel('time (mins)'), ylabel('infusions (last session)')
%
subplot(2,num_panels/num_rows,6)
hold on
plot(iii,'k--o')
set(gca,'xtick',0:5:30)
xlabel('infusion number'), ylabel('inter-infusion intervals (s)')
%
subplot(2,num_panels/num_rows,7)
hold on
plot(Xs(1:(30*60)/4,1),'k')
set(gca,'xtick',0:(10*60)/4:(30*60)/4,'xticklabel',0:10:30)
xlabel('time (mins)'), ylabel('internal state (last session)')
