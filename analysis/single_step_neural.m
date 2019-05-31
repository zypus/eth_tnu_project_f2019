function dxdt = single_step_neural(x, u, P) 
        % computes a single DCM update
    effective_connectivity = (P.A + u(2) * P.B + x(2) * P.D1 + x(3) * P.D2);
    effective_connectivity = effective_connectivity - diag(diag(effective_connectivity)) - 0.5 * diag(exp(diag(effective_connectivity)));
    %AuB(1,1) = -0.5 * exp(P.A(1,1) + u * P.B(1,1));
    %AuB(2,2) = -0.5 * exp(P.A(2,2) + u * P.B(2,2));

    dxdt = (effective_connectivity)*x+P.C*u;
end