function ljd = log_joint_distribution(data, U, P, P_hrf, x0, h0, sigma, param_mask, param_sigma)
    % Computes the log joint distribution log Pr(P|data)Pr(P) for the given model and data
    params = flatten_parameters(P);
    param_mask = flatten_parameters(param_mask);
    
    relevant_params = params.*param_mask;
    
    relevant_count = sum(param_mask);
    
    log_prior = -0.5 / param_sigma^2 * (relevant_params' * relevant_params) - relevant_count * log(param_sigma) - 0.5 * relevant_count * log(2 * pi);

    lp = log_likelihood(data, U, P, P_hrf, x0, h0, sigma);
       
    ljd = lp + log_prior; 
end