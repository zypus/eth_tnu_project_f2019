function lp = mcmc_log_likelihood(data, model, parname, parvalue, sigma)
    y = model(1, parname, parvalue);
    yDiff = (data(:)-y(:));
    
    n = size(yDiff,1);
    
    lp = -0.5 / sigma^2 * (yDiff' * yDiff) - n * log(sigma) - 0.5 * n * log(2 * pi); 
end
