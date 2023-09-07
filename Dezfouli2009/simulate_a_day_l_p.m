function [as, ss, r, r_e, r_bar, kappa, delta, Vs, Q, plt] = simulate_a_day_l_p(NT, Q, kappa_o, r_bar_o)

% length_sess = 1;

% Time variables
% t_step = 1080; % each point of time is 4 seconds
% NT = (24*60*60)/t_step; % number of time steps (24 hours)
% NTC = (length_sess*60*60)/t_step; % number of time steps in which cocaine is available

NTC = NT;

% Constants
Ds = 15;
alpha = 0.2;
lambd = 0.0003;
N = 2;
epsi = 0.1;
sigma = 0.005;
mu_N = 5;
sigma_N = 0.02;
mu_fr = -2;
sigma_fr = 0.02;
mu_sh = -200;
sigma_sh = 0.02;
mu_c = 2;
sigma_c = 0.02;
mu_s = 1;
sigma_s = 0.02;
mu_l = 15;
sigma_l = 0.02;

nS = 5;         % number of states
nA = 3;         % number of actions: 1=OTH (other), 2=ILP (inactive lever), 3=ALP (active lever)
OTH = 1;    ILP = 2;    ALP = 3;
cost = 1;       % cost of pressing the lever

ss = zeros(NT+1,1); ss(1) = 1; % external state
r = zeros(NT+1,1);       % reward r 
% r_c = zeros(NT+1,1);     % reward under cocaine influence
r_e = zeros(NT+1,1);      % experienced reward
r_bar = zeros(NT+1,1);  r_bar(1) = r_bar_o; % average reward at time t
as  = zeros(NT+1,1);     % actions taken
delta  = zeros(NT+1,1);  % reward prediciton error
% delta_c  = zeros(NT+1,1);  % reward prediciton error
% Q = zeros(nS,nA);       % pairs of state action values
kappa = zeros(NT+1,1);  kappa(1) = kappa_o; % abnormal baseline elevation
rho = zeros(NT+1,1);    rho(1) = kappa(1) + r_bar(1); % new r bar
Vs = zeros(NT+1,nS,nA);     % stores the value of drug event
plt = zeros(NT+1,1);

T = zeros(nS,nS,nA);        % true transition model 
T(:,:,OTH) = [1 0 0 0 0; 0 0 1 0 0; 0 0 0 1 0; 0 0 0 0 1; 1 0 0 0 0]'; % col=s(t), row=s(t+1)
T(:,:,ILP) = T(:,:,OTH);
T(:,:,ALP) = T(:,:,ILP);
T(:,1,ALP) = [0 1 0 0 0]';  % the only difference for ALP, is that choosing it in state 1 leads to transition to 2
R = zeros(nS, nA);          % Reward matrix

% sometimes the T and R matrixes are called the action-state graph
R(:,2:3) = -cost;

for t = 1:NT
        
%     if t>NTC
%         R(1,3) =  - cost; % if cocaine is no longer available
%     else
    R(1,3) = reward(mu_c,sigma_c) - cost;
    R(2,1) = 0;
    R(2,2:3) = -cost;
    
    if t>1 && ss(t-1) == 1 && as(t-1) == 3  % downside. The animal has taken the drug just before
        R(2,:) = R(2,:) - reward(mu_c,sigma_c);
    end
%     end

    % Choose the next action using the epsilon-greedy algorithm, and derive the next state
    
    exploratory = false;
    if numel(Q(ss(t),:)) >= 2 && rand() < epsi
        exploratory = true;
        as(t) = randi(numel(Q(ss(t), :)));
    else
        value_dict = zeros(1, nA);  % store estimated values of diferent actions in current sate
        for a = 1:nA
            value_dict(a) = Q(ss(t),a);
        end
        [max_val, max_idx] = max(value_dict);
        if length(value_dict) == length(find(value_dict == max_val))
            max_idx = randi(numel(Q(ss(t), :)));
        end 
        as(t) = max_idx;
    end

    % this line is not working
    ss(t + 1) = find(T(:, ss(t), as(t)));
    r(t) = R(ss(t), as(t));
    
    V = max(Q(ss(t+1),:));

    
    if ss(t) == 1 && as(t) == 3 
        kappa(t + 1) = (1 - lambd) * kappa(t) + lambd * N;
        delta(t) = max(R(ss(t), as(t)) + V - Q(ss(t), as(t)) + Ds - kappa(t), Ds - kappa(t)) - r_bar(t);
%     elseif t>1 && ss(t-1) == 1 && as(t-1) == 3 
%         kappa(t + 1) = (1 - lambd) * kappa(t);
%         delta(t) = 0.2*(min(R(ss(t), as(t)) + V - Q(ss(t), as(t)) - Ds - kappa(t), -Ds - kappa(t)) - r_bar(t));
    else
        kappa(t + 1) = (1 - lambd) * kappa(t);
        delta(t) = R(ss(t), as(t)) + V - Q(ss(t), as(t)) - kappa(t) - r_bar(t);
    end
    
    Q(ss(t),as(t)) = Q(ss(t),as(t)) + delta(t) * alpha; % learning is proportional to delta
    Vs(t,:,:) = Q(:,:);
    
    r_e(t) = delta(t) - V + Q(ss(t), as(t)) + r_bar(t) + kappa(t);  % only want to compute it in order to update r_bar
                           
    % store this quantity
    plt(t) = -V + Q(ss(t), as(t));
    
    if exploratory  % average reward is computed over nonexplanatory actions
        r_bar(t + 1) = r_bar(t);
    else
        r_bar(t + 1) = (1 - sigma) * r_bar(t) + sigma * r_e(t);
    end
end
    

% instead of having 2 deltas I should only have 1!!!!
% in the same way, only one experienced reward!


