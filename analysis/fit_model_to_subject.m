function fit = fit_model_to_subject(model, sub)
%FIT_MODEL_TO_SUBJECT Given a model and a subject tries to fit the model
%using bayesian inference
% OUTPUT
%  fit = sub
% + fit.F - Negative Free Energy = log model evidence

    U.u = sub.u;
    U.dt = 0.1;
    U.subsample = 20;

    P = load_model("model" + model);

    mask = structure_parameters(flatten_parameters(P) ~= 0, P);

    P_hrf.kappa = 0.64;
    P_hrf.gamma = 0.32;
    P_hrf.tau = 2;
    P_hrf.alpha = 0.32;
    P_hrf.E0 = 0.4;
    
    x0 = zeros(size(sub.y,1),1);
    h0 = [0, 1, 1, 1]';

    M.IS = @(p,M,U) euler_integrate_dcm_y(U, merge_parameters(P, mask, p), P_hrf, x0, h0)';
    M.pE = zeros(sum(flatten_parameters(mask)),1);
    M.pC = eye(sum(flatten_parameters(mask)));
    
    %M.hE = 1;
    %M.hC = 1000;
    %M.P = P;
    
    M.nograph = 1;
    M.Nmax = 30;
    
    Y.y = sub.y';
    Y.dt = 2;
    %Y.X0 = sub.x0;
    %Y.Q = stdY;

    [Ep, ~, ~, F] = spm_nlsi_GN(M,U,Y); % NOTE: should not have discared Cp here, because else I could also habe done BMA

    sub.P = merge_parameters(P, mask, Ep);
    
    sub.F = F;
    
    fit = sub;
end

