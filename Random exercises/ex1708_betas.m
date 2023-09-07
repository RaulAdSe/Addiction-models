% Define the range of ratios to explore
% Store nothing and test the effect on the ratio of betas in the final r_bar.

ratio_range = 0.05:0.05:0.5; % You can adjust the range as needed
ratio_range = [ratio_range 0.5:0.1:2];
epsi = 1e-2;
NT = 24 * 1e4;
N = 24;
N_C = 6;
nA = 2;
nS = N;
T = zeros(nS, nS, nA);
for i = 1:N
    for j = 1:N-1
        if i == j
            T(i+1,j,:) = 1;
        end
    end
end
T(1,end,:) = 1;
R = zeros(nS, nA);
for i = 1: N_C
    R(i,1) = 1;
end


% Initialize an array to store r_bar end values
r_bar_end_values = zeros(size(ratio_range));

for i = 1:length(ratio_range)
    % Set beta_minus and beta_plus based on the current ratio
    beta_ratio = ratio_range(i);
    beta_plus = 1e-2;
    beta_minus = beta_plus*beta_ratio;
    
    ss = 1;
    r_bar = 0;
    Q = zeros(nS,nA);

    for t = 1:NT

        exploratory = false;
        if rand() < epsi
            exploratory = true;
            as = randi(numel(Q(ss, :)));
        else
            value_dict = zeros(1, nA);  % store estimated values of diferent actions in current sate
            for a = 1:nA
                value_dict(a) = Q(ss,a);
            end
            [~, max_idx] = max(value_dict);
            as = max_idx;
        end

        ss_next = find(T(:, ss, as));
        r = R(ss, as);

        V = max(Q(ss_next,:));

        delta = r + V - Q(ss, as) - r_bar;

        if delta > 0
            beta = beta_plus;
            alpha = alpha_plus;
        else
            beta = beta_minus;
            alpha = alpha_minus;
        end

        r_bar = r_bar + beta * delta;

        if ss == N   % Fixed value state
            Q(ss,:) = 0;
        else
            Q(ss,as) = Q(ss,as) + alpha * delta;
        end
        
        ss = ss_next;
    end

    % Run the TDRL model

    % Store the last value of r_bar
    r_bar_end_values(i) = r_bar;
end

% Plot the results
figure;
plot(ratio_range, r_bar_end_values, 'o-');
xlabel('Betas ratio');
ylabel('r_{bar}*');
grid on;
