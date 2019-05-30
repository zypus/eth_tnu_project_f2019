function output = standardize(input)
%NORMALIZE Summary of this function goes here
%   Detailed explanation goes here
    mu = mean(input);
    sigma = std(input);
    
    if sigma > 0.00001 
        output = (input - mu) ./ sigma;
    else
        output = input - mu;
    end
end

