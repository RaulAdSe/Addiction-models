% Reward function
function val = reward(mu, sigma)
    val  = mu + sigma * randn();
end
