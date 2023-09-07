N = 1;
nA = 2;
nS = (2)*N + 2;
T = zeros(nS, nS, nA);
T(:, :, 1) = [0 1 0 0; 0 0 0 1;0 0 0 1; 1 0 0 0]';
T(:, :, 2) = [0 0 1 0; 0 0 0 1;0 0 0 1; 1 0 0 0]';
R = zeros(nS, nA);
R(1,1) = 1;

NT = 100000;

ss = zeros(NT + 1, 1);
ss(1) = 1;
r = zeros(NT, 1);
r_bar = zeros(NT + 1, 1);
as = zeros(NT + 1, 1);
delta = zeros(NT + 1, 1);
Q = zeros(nS,nA);
Vs = zeros(NT + 1, nS,nA);

% Ds = 15;
beta = 1e-3; 
alpha = 1e-1;
epsi = 5e-2;


alpha_values = linspace(0.01, 1, 10); % Modify the range of alpha values
beta_values = linspace(0.01, 1, 10);  % Modify the range of beta values

% Initialize a matrix to store the convergence results
convergence_matrix = zeros(length(alpha_values), length(beta_values));

% Loop through different combinations of alpha and beta values
for alpha_idx = 1:length(alpha_values)
    for beta_idx = 1:length(beta_values)
        alpha = alpha_values(alpha_idx);
        beta = beta_values(beta_idx);
        
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

            ss(t + 1) = find(T(:, ss(t), as(t)));
            r(t) = R(ss(t), as(t));

            V = max(Q(ss(t+1),:));

            delta(t) = r(t) + V - Q(ss(t), as(t)) - r_bar(t);

            r_bar(t + 1) = r_bar(t) + beta * delta(t);

            if ss(t) == 4   % Fixed value state
                Q(ss(t),:) = 0;
                    % The agent is not learning well the Q-values! Delta should
                    % tend to 0 in this state as well.
            else
                Q(ss(t),as(t)) = Q(ss(t),as(t)) + alpha * delta(t);
            end

            Vs(t,:,:) = Q(:,:); % fix one so the others converge
        end
        
        % The delta of the state that I have fixed will have a slower timescale
        s4_indx = find(ss == 4);
        s4_indx_last = s4_indx(end-9:end);
        recent_deltas = abs(delta(s4_indx_last));
        if any(recent_deltas < 1e-2)
            convergence_matrix(alpha_idx, beta_idx) = 1;
            break;  % Exit the loop early if converged
        end
    end
end      

% Display the convergence matrix
disp('Convergence Matrix:');
disp(convergence_matrix);