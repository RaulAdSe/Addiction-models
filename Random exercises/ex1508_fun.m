function [ss,as,r,delta,Vs,r_bar] = ex1508_fun(NTC,r_bar_0,Q_0)

N = 1;
nA = 2;
nS = (2)*N + 2;
T = zeros(nS, nS, nA);
T(:, :, 1) = [0 1 0 0; 0 0 0 1;0 0 0 1; 1 0 0 0]';
T(:, :, 2) = [0 0 1 0; 0 0 0 1;0 0 0 1; 1 0 0 0]';
R = zeros(nS, nA);
R(1,1) = 1;

NT = 24000;

ss = zeros(NT + 1, 1);
ss(1) = 1;
r = zeros(NT + 1, 1);
r_bar = zeros(NT + 1, 1);
r_bar(1) = r_bar_0;
as = zeros(NT + 1, 1);
delta = zeros(NT + 1, 1);
Q = zeros(nS,nA);
Q = Q_0;
Vs = zeros(NT + 1, nS,nA);

beta_plus = 1e-3; 
beta_minus = 1e-3; 
alpha_plus = 1e-2;
alpha_minus = 1e-1;

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
        cum_rew = sum(r(:));
    end
    
    ss(t + 1) = find(T(:, ss(t), as(t)));
    r(t) = R(ss(t), as(t));

    V = max(Q(ss(t+1),:));

    delta(t) = r(t) + V - Q(ss(t), as(t)) - r_bar(t);
    
    if t>NTC && indx == 0 && as(t) == 1 && ss(t) == 1
        delta(t) = r(t) + V - Q(ss(t), as(t)) - r_bar(t) - m*cum_rew;
        indx = 1;
    end
    
    if delta(t) > 0
        beta = beta_plus;
        alpha = alpha_plus;
    else
        beta = beta_minus;
        alpha = alpha_minus;
    end

    
    r_bar(t + 1) = r_bar(t) + beta * delta(t);

    if ss(t) == 4   % Fixed value state
        Q(ss(t),:) = 0;
    else
        Q(ss(t),as(t)) = Q(ss(t),as(t)) + alpha * delta(t);
    end

    Vs(t,:,:) = Q(:,:); % fix one so the others converge
end