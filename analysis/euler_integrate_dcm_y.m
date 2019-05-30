function y = euler_integrate_dcm_y(U, P, P_hrf, x0, h0)
%EULER_INTEGRATE_DCM_Y Summary of this function goes here
%   Detailed explanation goes here
    [y, ~, ~] = euler_integrate_dcm(U, P, P_hrf, x0, h0);
    
    if isfield(U, "subsample")
        y = y(:,1:U.subsample:end);
    end
    
end

