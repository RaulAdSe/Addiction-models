clear; clc; close all;

nS = 5;
nA = 2;

days = 100;
NT = 500;

% Define parameter ranges to explore
NTC_range = [10, 25, 50, 100, 350]; % Vary NTC values
beta_plus_range = linspace(1e-4,1e-2,5); % Vary beta_plus values
beta_minus_range = linspace(1e-4,1e-2,5); % Vary beta_minus values

% Create arrays to store results
results = zeros(length(NTC_range), length(beta_plus_range), length(beta_minus_range));

% Loop through parameter combinations
for i = 1:length(NTC_range)
    for j = 1:length(beta_plus_range)
        for k = 1:length(beta_minus_range)
            NTC = NTC_range(i);
            beta_plus = beta_plus_range(j);
            beta_minus = beta_minus_range(k);
            
            % Simulate the model and capture relevant information
            Q_0 = zeros(nS,nA);       % pairs of state action values
            r_bar_0 = 0;

            as_all = [];
            ss_all = [];
            r_all = [];
            r_bar_all = [];
            delta_all = [];
            Vs_all = zeros(NT*days,nS,nA);

            for kk = 1:days
                    [ss,as,r,delta,Vs,r_bar] = ex1808_fun(NTC,r_bar_0,Q_0,beta_plus,beta_minus);
%                     as_all = [as_all as(1:end-1)];
%                     ss_all = [ss_all ss(1:end-1)];
                    r_all = [r_all r(1:end-1)];
%                     r_bar_all = [r_bar_all r_bar(1:end-1)];
%                     Vs_all(1 + (kk-1)*NT:kk*NT,:,:) = Vs(1:end-1,:,:);
%                     delta_all = [delta_all delta(1:end-1)];

                    r_bar_0 = r_bar(end-1);
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
            for chunk_idx = 1:num_chunks-1
                start_idx = (chunk_idx - 1) * chunk_size + 1;
                end_idx = chunk_idx * chunk_size;
                % Count the number of ones in the current chunk
                chunk_counts(chunk_idx) = sum(r_c(start_idx:end_idx));
            end
            
%             figure()
%             bar(chunk_counts);
%             ylabel('Number of Ones');
%             title_str = sprintf('Number of Rewards each Day (N_C = %d, \\beta_+ = %.4f, \\beta_- = %.4f)', ...
%                 NTC_range(i), beta_plus_range(j), beta_minus_range(k));
%             title(title_str);            
%             pause(0.5)
%             
            % Store the result in the results array
            results(i, j, k) = mean(chunk_counts); % You can also use other metrics
        end
    end
end

% Visualize the results
figure;
for i = 1:length(NTC_range)
    subplot(length(NTC_range), 1, i);
    A = zeros(length(beta_plus_range),length(beta_plus_range));
    for j = 1:length(beta_plus_range)
        for k = 1:length(beta_plus_range)
            A (j,k) = results(i, j, k);
        end
    end
    
    imagesc(beta_plus_range, beta_minus_range, A);
    grid on
    colorbar; % Add colorbar
    if i == 1
        cb = colorbar;
        cb.Label.String = 'Mean daily intake';
    end
    xlabel('\beta_{plus}');
    ylabel('\beta_{minus}');
    title(['Cocaine availability = ' num2str(NTC_range(i)/NT * 100) ' %']);
end
