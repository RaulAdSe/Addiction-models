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

% Transition and Reward Matrices
nS = 1;         % number of states
nA = 1;         % number of actions: 1=OTH (other), 2=ILP (inactive lever), 3=ALP (active lever)

T = zeros(nS, nS, nA);
T(:, :, 1) = [1];
% T(:, :, 2) = [0, 1, 0; 0, 1, 0; 0, 1, 0];
% T(:, :, 3) = [0, 0, 1; 1, 0, 0; 0, 0, 1];
R = zeros(nS, nA);
R(1) = reward(mu_c,sigma_c);

% Variables
% NT = 24 * 60 * 60 / t_step;
NT = 2000;

% NTC = T * 60 * 60 / t_step;
ss = zeros(NT + 1, 1);
ss(1) = 1;
r = zeros(NT + 1, 1);
r_c = zeros(NT + 1, 1);
r_bar = zeros(NT + 1, 1);
r_bar(1) = 0;
as = zeros(NT + 1, 1);
delta = zeros(NT + 1, 1);
delta_c = zeros(NT + 1, 1);
Q = zeros(nS,nA);
kappa = zeros(NT + 1, 1);
kappa(1) = 0;
rho = zeros(NT + 1, 1);
Vs = zeros(NT + 1, 1);
% Simulation
for t = 1:NT

    exploratory = false;
    if numel(Q(ss(t))) >= 2 && rand() < epsi
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

    [~, ss(t + 1)] = find(T(:, ss(t), as(t)));
    r(t) = R(ss(t), as(t));
    
    V = max(Q(ss(t+1)));
    
    kappa(t + 1) = (1 - lambd) * kappa(t) + lambd * N;  % since it is independent
    
    rho(t) = r_bar(t) + kappa(t);
    
    delta_c(t) = max(R(ss(t), as(t)) + V - Q(ss(t), as(t)) + Ds - kappa(t), Ds - kappa(t)) - r_bar(t);
        
    Q(ss(t),as(t)) = Q(ss(t),as(t)) + delta_c(t) * alpha;
    
    r_c(t) = delta_c(t) - V + Q(ss(t), as(t)) + rho(t);
    

    if exploratory
        r_bar(t + 1) = r_bar(t);
    else
        r_bar(t + 1) = (1 - sigma) * r_bar(t) + sigma * r_c(t);
    end
    

    Vs(t) = V;
end

figure()
plot([1:NT],Vs(1:end-1));
figure()
plot([1:NT],r_bar(1:end-1));
