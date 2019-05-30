function sim = simulate_new_data_for_subject(subject, model, sigma)
%SIMULATE_NEW_DATA_FOR_SUBJECT Summary of this function goes here
%   Detailed explanation goes here
    sub = load_subject(subject);
    
    U.u = sub.u;
    U.dt = 0.1;
    U.subsample = 20;

    P = load_model("model" + model);
    
    flat_P = flatten_parameters(P);
    
    chars = char(subject);
    hash = int32(sum(double(chars).*(10.^(1:length(chars)))));
    
    rng(hash);
    
    mask = flat_P ~= 0;
    
    % add some noise to the parameteres
    P = structure_parameters((flat_P + sigma * randn(size(flat_P))).*mask, P);

    P_hrf.kappa = 0.64;
    P_hrf.gamma = 0.32;
    P_hrf.tau = 2;
    P_hrf.alpha = 0.32;
    P_hrf.E0 = 0.4;
    
    x0 = zeros(size(sub.y,1),1);
    h0 = [0, 1, 1, 1]';

    sim = sub;
    
    [y, ~, ~] = euler_integrate_dcm(U, P, P_hrf, x0, h0);
    
    stdY = mean(std(y, 0, 2));
    y_head = y + stdY * randn(size(y));

    sim.y = y_head(:,1:U.subsample:end);
    
end

