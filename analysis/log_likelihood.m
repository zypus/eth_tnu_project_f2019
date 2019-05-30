function lp = log_likelihood(data, U, P, P_hrf, x0, h0, sigma)
    [y, ~, ~] = euler_integrate_dcm(U, P, P_hrf, x0, h0);
    if isfield(U, "subsample")
        y = y(:,1:U.subsample:end);
    end
    yDiff = (data(:)-y(:));
    
    n = size(yDiff,1);
    
    lp = -0.5 / sigma^2 * (yDiff' * yDiff) - n * log(sigma) - 0.5 * n * log(2 * pi); 
end