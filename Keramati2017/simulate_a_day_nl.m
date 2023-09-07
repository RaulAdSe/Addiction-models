function [XXs, Xs, as, ss] = simulate_a_day_nl( length_sess, XX_0, X_0 )

% NO LEARNING

% simulate a 24 hour period; it begins when the animal's daily session
% starts
% INPUTS
% length_sess: length of session (i.e., cocaine access) in hours
% XX_0: initial set point 
% X_0: initial internal state h

t_step = 4; % each point of time is 4 seconds
NT = (24*60*60)/t_step; % number of time steps (24 hours)
NTC = (length_sess*60*60)/t_step; % number of time steps in which cocaine is available

% parameters
ds = 3; % depth of planning (i.e., number of expansions)
gamm = .999; % discount rate
m = 3;
n = 4;
thr_min = 100;
thr_max = 200; 
K = 50; % effect of 0.25mg of cocaine on internal state
phi = 7e-3; % rate of elimination of cocaine
omega = 12e-2; % rate of absorption of cocaine
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

% transition model (same for both MDPs)
T = zeros(nS,nS,nA);
T(:,:,OTH) = [1 0 0 0 0; 0 0 1 0 0; 0 0 0 1 0; 0 0 0 0 1; 1 0 0 0 0]'; % col=s(t), row=s(t+1)
T(:,:,ILP) = T(:,:,OTH);
T(:,:,ALP) = T(:,:,ILP);
T(:,1,ALP) = [0 1 0 0 0]'; % the only difference for ALP, is that choosing it in state 1 leads to transition to 2

% immediate (expected) 'non-homeostatic' rewards; this is just the cost of pressing
% the lever for ILP and ALP
R_nh = zeros(nS,nA);
R_nh(:,[ILP ALP]) = -c; % cost of a lever press (note above that we set c>=0)

% the game is, at each time point, to do a 3-step look ahead to construct
% value, with the approximation that (h,h*) are fixed
V = zeros(nS,ds+1); % use to cache values during planning
Q = zeros(nS,nA);
X = X_0; 
XX = XX_0; 
cc = 0; % 'cocaine buffer'
s = 1; % 'external' state
dummy_vec = [1 0 0 0 0]'; % state 1 is special, since it's the only state you can press and get cocaine (i.e., cocaine lever is active)
% recordings
Xs = zeros(NT+1,1); Xs(1) = X_0;
XXs = zeros(NT+1,1); XXs(1) = XX_0;
ccs = zeros(NT+1,1); ccs(1) = 0; % cocaine buffer
ss = zeros(NT+1,1); ss(1) = 1; % external state
as = zeros(NT,1); % actions taken
for i = 1:NT
    
    if i>NTC
        K=0; % if cocaine is no longer available
    end
    
    % each time step involves planning; not that during planning, it is
    % assumed that current state [h(t),h*(t)] is FIXED (there is not forward simulation of the dynamics of these)
    V(:)=0;
    for j = ds:-1:1
        Q(:,:)=0;
        for k = 1:nA
            Xp = X+K.*(k==ALP).*dummy_vec; % I believe this is what Mehdi uses for projected next internal state: either the same or a step change due to cocaine
%             Xp = (1-phi)*X + omega*(cc+K.*(k==ALP).*dummy_vec); %
%             projected next internal state if one were using more dynamics (i.e., degradation of any cocaine in the system + absorption)
            R_h = ((XX-X)^n)^(1/m) - ((XX-Xp).^n).^(1/m); % 'homeostatic' reward (or what Mehdi calls 'drive-reduction reward')
            Q(:,k) = (R_nh(:,k) + R_h) + gamm.*T(:,:,k)'*V(:,j+1);
        end
        V(:,j) = max(Q,[],2);
    end
    % so the last Q-function that we get here tells us which action to execute, given
    % our current external state
    [~,a] = max(Q(s,:));
    
    % given action execution, actual updates
    XX = XX - rho + mu*(s==1 & a==ALP)*K; % set point (only moves upward if take cocaine)
    XX = max(XX,thr_min);
    XX = min(XX,thr_max); 
    X = (1-phi)*X + omega*cc; % state variable: I believe this is what Mehdi uses -- changes in X only via buffer
%     X = (1-phi)*X + omega*(cc+K.*(s==1 & a==ALP)); % state variable
    cc = (1-omega)*cc + (s==1 & a==ALP)*K; % 'cocaine buffer'
    s = find( T(:,s,a) ); % external state; since dynamics are deterministic, just find the entry with the '1'
    
    % record
    as(i) = a;
    ss(i+1) = s;
    Xs(i+1) = X;
    XXs(i+1) = XX;
    ccs(i+1) = cc;
    
end

ss = ss(1:end-1);
Xs = Xs(1:end-1);
XXs = XXs(1:end-1);
ccs = ccs(1:end-1);

end