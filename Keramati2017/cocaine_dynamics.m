% we want to understand how the dynamics of cocaine administration look.
% we believe this is what Mehdi used to simulate. (cf. Figure 1 of psych
% review paper).

close all; clear; clc

T = (50*60)/4; % number of time steps (each is 4s)
t_infusions = [100 200]; % on which timesteps we assume any ADDITIONAL cocaine is taken
K = 50; % let's say this is a constant resulting from a unit dose of cocaine
cc = 50; % so this is the initial amount in the 'cocaine buffer'
omega = 12e-2; % absorption rate from buffer
phi = .7e-2;   % elimination rate

% dynamics, discrete process
t_c = zeros(1,T); t_c(t_infusions)=1;
h = zeros(T,1);
cc = zeros(T,1); cc(1) = K;
for i=1:T-1
   h(i+1) = h(i) + omega*cc(i) - phi*h(i); % (1-phi)*h(i) + omega*cc(i); % h is producedas an effect off
   cc(i+1) = cc(i) - omega*cc(i) + t_c(i)*K; %(1-omega)*cc(i) + t_c(i)*K;
end

%plotting
plot([4:4:T*4]./60, h,'r')
hold on
plot([4:4:T*4]./60, cc,'g')
legend({'h = cocaine in brain','cocaine in body'})
plot(linspace(0,(T*4)/60), zeros(1,100), 'k--')
xlabel('time (min)'), ylabel('h (cocaine-related variable)')
set(gca,'xtick',0:10:40, 'xlim', [-1 (T*4)/60])