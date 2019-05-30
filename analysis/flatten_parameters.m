function [pf] = flatten_parameters(P)
%FLATTEN_PARAMETERS Summary of this function goes here
%   Detailed explanation goes here

 At = P.A;
 Bt = P.B;
 Ct = P.C;
 D1t = P.D1;
 D2t = P.D2;

 pf = [At(:);Bt(:);Ct(:);D1t(:);D2t(:);P.kappa;P.gamma;P.tau;P.alpha;P.E0];

end

