function y = compute_bold_signal(h, P_hrf)
    V0 = 0.04;

    v0 = 80.6;
    r0 = 110;
    TE = 0.035;
    eps = 0.47;
    
    k1 = 4.3*v0*P_hrf.E0*TE;
    k2 = eps*r0*P_hrf.E0*TE;
    k3 = 1-eps;
    
    y = V0 * (k1 * (1-h(4)) + k2*(1-(h(4)/h(3))+ k3*(1-h(3))) );
end