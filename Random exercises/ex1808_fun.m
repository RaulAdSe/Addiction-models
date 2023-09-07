function [ss,as,r,delta,Vs,r_bar] = ex1808_fun(N_C,r_bar_0,Q_0,beta_plus,beta_minus, alpha_plus, alpha_minus)

    NT = 500;
%     N_C = 1000;

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
    r_bar(1) = r_bar_0;
    as = zeros(NT + 1, 1);
    delta = zeros(NT + 1, 1);
%     Q = zeros(nS,nA);
    Q = Q_0;
    Vs = zeros(NT + 1, nS,nA);

%     beta_plus = 4e-4; 
%     beta_minus = 1e-3; 

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
end

