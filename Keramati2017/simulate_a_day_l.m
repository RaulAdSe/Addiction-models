function [XXs, Xs, bbel, as, ss, T_hat_nc, O_hat_nc, T_hat_c, O_hat_c, Vs] =...
    simulate_a_day_l( length_sess, XX_0, X_0, c_0, T_hat_nc, O_hat_nc, T_hat_c, O_hat_c )

% LEARNING: including Mehdi's learning mechanism for transition and outcome
% functions; let's also throw in softmax selection for good measure

% simulate a 24 hour period; it begins when the animal's daily session
% starts
% INPUTS
% length_sess: length of session (i.e., cocaine access) in hours
% XX_0: initial set point 
% X_0: initial internal state h
% c_0: initial 'belief state'
% T_hat_nc: current estimated transition model, cocaine-unavailable (nS x nS x nA)
% O_hat_nc: current estimated outcome model, cocaine-unavailable (nS x nA)
% T_hat_c: current estimated transition model, cocaine available (nS x nS x nA)
% O_hat_c: current estimated outcome model, cocaine available (nS x nA)

t_step = 4; % each point of time is 4 seconds
NT = (24*60*60)/t_step; % number of time steps (24 hours)
NTC = (length_sess*60*60)/t_step; % number of time steps in which cocaine is available

% parameters
lr = .2; % learning rate for outcome and transition functions
tau = .25; % temperature of softmax
ds = 3; % depth of planning (i.e., number of expansions)
gamm = 1; % discount rate
m = 3;
n = 4;
thr_min = 100;
thr_max = 200; 
K = 50; % effect of 0.25mg of cocaine on internal state
phi = 7e-3; % rate of elimination of cocaine
omega = 11.3e-2; % rate of absorption of cocaine
% omega = 11.3e-2; % rate of absorption of cocaine
mu = 0.0018; % rate of setpoint up-regulation (if cocaine)
rho = 0.00016; % rate of setpoint recovery
c = 1; % energy cost of a lever press

% things common to both MDPs
nS = 5; % number of states, each occupied for 4 seconds = 20s
nA = 3; % number of actions: 1=OTH (other), 2=ILP (inactive lever), 3=ALP (active lever)
OTH = 1;
ILP = 2;
ALP = 3;

% true transition model 
T = zeros(nS,nS,nA);
T(:,:,OTH) = [1 0 0 0 0; 0 0 1 0 0; 0 0 0 1 0; 0 0 0 0 1; 1 0 0 0 0]'; % col=s(t), row=s(t+1)
T(:,:,ILP) = T(:,:,OTH);
T(:,:,ALP) = T(:,:,ILP);
T(:,1,ALP) = [0 1 0 0 0]'; % the only difference for ALP, is that choosing it in state 1 leads to transition to 2

% immediate (expected) 'non-homeostatic' rewards; this is just the cost of pressing
% the lever for ILP and ALP (I think Mehdi just assumes the animal knows this -- i.e., doesn't need to be learned)
R_nh = zeros(nS,nA);
R_nh(:,[ILP ALP]) = -c; % cost of a lever press (note above that we set c>=0)

% the game is, at each time point, to do a 3-step look ahead to construct
% value, with the approximation that (h,h*) are fixed
V_nc = zeros(nS,ds+1); % use to cache values during planning
Q_nc = zeros(nS,nA);
V_c = zeros(nS,ds+1); % use to cache values during planning
Q_c = zeros(nS,nA);
X = X_0; 
XX = XX_0; 
bel = c_0; % belief state
cc = 0; % 'cocaine buffer'
s = 1; % 'external' state
dummy_vec = [1 0 0 0 0]'; % state 1 is special, since it's the only state you can press and get cocaine (i.e., cocaine lever is active)
% recordings
Xs = zeros(NT+1,1); Xs(1) = X_0;
XXs = zeros(NT+1,1); XXs(1) = XX_0;
ccs = zeros(NT+1,1); ccs(1) = 0; % cocaine buffer
ss = zeros(NT+1,1); ss(1) = 1; % external state
as = zeros(NT,1); % actions taken
bbel = zeros(NT+1,1); bbel(1) = c_0; % beliefs
Vs = zeros(NT+1,1); % drug value
for i = 1:NT
    
    if i>NTC
        K=0; % if cocaine is no longer available
    end
    
    % each time step involves planning; not that during planning, it is
    % assumed that current state [h(t),h*(t)] is FIXED (there is not forward simulation of the dynamics of these)
    V_nc(:)=0;
    V_c(:)=0;
    for j = ds:-1:1
        Q_nc(:,:)=0;
        Q_c(:,:)=0;
        for k = 1:nA
            Xp_nc = X + O_hat_nc(:,k); % no cocaine MDP. potential next internal state
            Xp_c = X + O_hat_c(:,k);
            R_h_nc = ((XX-X)^n)^(1/m) - ((XX-Xp_nc).^n).^(1/m); 
            R_h_c = ((XX-X)^n)^(1/m) - ((XX-Xp_c).^n).^(1/m); % 'homeostatic' reward (or what Mehdi calls 'drive-reduction reward')
            Q_nc(:,k) = (R_h_nc + R_nh(:,k)) + gamm.*T_hat_nc(:,:,k)'*V_nc(:,j+1);
            Q_c(:,k) = (R_h_c + R_nh(:,k)) + gamm.*T_hat_c(:,:,k)'*V_c(:,j+1);
        end
        V_nc(:,j) = max(Q_nc,[],2);
        V_c(:,j) = max(Q_c,[],2);
    end
    % so the last Q-function that we get here tells us which action to execute, given
    % our current external state
    Qtemp = bel.*Q_nc(s,:) + (1-bel).*Q_c(s,:);
    
    % We want to store the Q value of the drug event
    Vs(i) = Qtemp(1,3);
    
    ps = exp( Qtemp./tau );
    ps = ps./sum(ps);
    a = 1 + sum(rand(1)>cumsum(ps)); % select action according to softmax
    
    % given action execution, actual updates
    XX = XX - rho + mu*(s==1 & a==ALP)*K; % set point (only moves upward if take cocaine)
    XX = max(XX,thr_min);
    XX = min(XX,thr_max); 
%     X = (1-phi)*X + omega*cc; % state variable: I believe this is what Mehdi uses -- changes in X only via buffer
%     X = (1-phi)*X + omega*(cc+K.*(s==1 & a==ALP)); % state variable
    X = X + omega*cc - phi*X; % state variable: I believe this is what Mehdi uses -- changes in X only via buffer
%     cc = (1-omega)*cc + (s==1 & a==ALP)*K; % 'cocaine buffer'
    cc = cc - omega*cc + (s==1 & a==ALP)*K; % 'cocaine buffer'

    s_new = find( T(:,s,a) ); % external state; since dynamics are deterministic, just find the entry with the '1'
    bel_new = min(1,X/XX); % new belief state
    
    % learning
    O_hat_c(s,a) = (1-lr*bel)*O_hat_c(s,a) + (lr*bel)*(s==1 && a==ALP)*K; 
    O_hat_nc(s,a) = (1-lr*(1-bel))*O_hat_c(s,a) + (lr*(1-bel))*(s==1 && a==ALP)*K;

    T_hat_c(s,s_new,a) = (1-lr*bel)*T_hat_c(s,s_new,a) + lr*bel; % basically just bumping up the count
    T_hat_nc(s,s_new,a) = (1-lr*(1-bel))*T_hat_c(s,s_new,a) + lr*(1-bel);
  
    s = s_new;
    bel = bel_new;
    
    % record
    as(i) = a;
    ss(i+1) = s;
    Xs(i+1) = X;
    XXs(i+1) = XX;
    ccs(i+1) = cc;
    bbel(i+1) = bel;
    
end

ss = ss(1:end-1);
Xs = Xs(1:end-1);
XXs = XXs(1:end-1);
bbel = bbel(1:end-1);

end