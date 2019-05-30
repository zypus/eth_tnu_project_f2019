function dhdt = single_step_hrf(h, u, P, P_hrf)

    kappa = exp(P.kappa) * P_hrf.kappa;
    tau = exp(P.tau) * P_hrf.tau;
    gamma = exp(P.gamma) * P_hrf.gamma;
    alpha = exp(P.alpha) * P_hrf.alpha;
    E0 = exp(P.E0) * P_hrf.E0;

    dhdt = zeros(4, 1);
    
    dhdt(1) = u - kappa * h(1) - gamma  * (h(2) - 1);
    dhdt(2) = h(1);
    dhdt(3) = 1/tau * (h(2) - h(3)^(1/alpha));
    dhdt(4) = 1/tau * (h(2) * (1-(1-E0)^(1/h(2)))/E0 - h(3)^(1/alpha) * h(4) / h(3));
end