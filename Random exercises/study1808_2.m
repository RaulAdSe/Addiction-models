% Define fixed values
beta_minus = 1e-3;
alpha_minus = 1e-1;

days = 100;
NT = 500;
nS = 5;
nA = 2;

% Calculate the ratio between alpha_plus and alpha_minus
NTC_range = [10, 25, 50, 100, 350]; % Vary NTC values
alpha_ratio = linspace(1e-3, 1e-1, 10); % Vary the ratio from 0.01 to 2
beta_ratio = linspace(1e-1, 1e1, 10); % Set beta_ratio equal to alpha_ratio

% Create arrays to store results
results = zeros(length(alpha_ratio),length(alpha_ratio));

% Loop through parameter combinations
for k = 1:length(NTC_range)
    for i = 1:length(alpha_ratio)
        for j = 1:length(alpha_ratio)
            NTC = NTC_range(k);
            alpha_plus = alpha_ratio(i) * alpha_minus;
            beta_plus = beta_ratio(j) * beta_minus;

            % Simulate the model and capture relevant information
            Q_0 = zeros(nS, nA); % pairs of state-action values
            r_bar_0 = 0;

            r_all = [];

            for kk = 1:days
                [ss,as,r,delta,Vs,r_bar] = ex1808_fun(NTC,r_bar_0,Q_0,beta_plus,beta_minus,alpha_plus,alpha_minus);
                r_all = [r_all r(1:end-1)];

                r_bar_0 = r(end-1);
                Q_0(1,:) = Vs(end-1,1,:);
                Q_0(2,:) = Vs(end-1,2,:);
            end

            r_c = r_all(:);
            % Set the chunk size
            chunk_size = NT;
            % Calculate the number of chunks
            num_chunks = numel(r_c) / chunk_size;
            % Initialize a vector to store the counts
            chunk_counts = zeros(round(num_chunks), 1);
            % Loop through the chunks
            for chunk_idx = 1:num_chunks - 1
                start_idx = (chunk_idx - 1) * chunk_size + 1;
                end_idx = chunk_idx * chunk_size;
                % Count the number of ones in the current chunk
                chunk_counts(chunk_idx) = sum(r_c(start_idx:end_idx));
            end

            % Store the result in the results array
            results(k,i,j) = mean(chunk_counts); % You can also use other metrics
        end
    end
end

figure;
for i = 1:length(NTC_range)
    subplot(length(NTC_range), 1, i);
    A = zeros(length(alpha_ratio),length(alpha_ratio));
    for j = 1:length(alpha_ratio)
        for k = 1:length(alpha_ratio)
            A (j,k) = results(i, j, k);
        end
    end
    imagesc(alpha_ratio, beta_ratio, A);
    grid on
    colorbar; % Add colorbar
    if i == 1
        cb = colorbar;
        cb.Label.String = 'Mean daily intake';
    end
    xlabel('alpha ratio')
    ylabel('beta ratio')
%     xlabel('\frac{\alpha_{-},\alpha_{+}}');
%     ylabel('\frac{\beta_{-},\beta_{+}}');
    title(['Cocaine availability = ' num2str(NTC_range(i)/NT * 100) ' %']);
end
