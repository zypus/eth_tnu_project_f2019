function [pf] = flatten_parameters(P)
%FLATTEN_PARAMETERS Takes a DCM parameter structure and collects all values
%into and array (inverse of STRUCTURE_PARAMETERS)

 At = P.A;
 Bt = P.B;
 Ct = P.C;
 D1t = P.D1;
 D2t = P.D2;

 pf = [At(:);Bt(:);Ct(:);D1t(:);D2t(:);P.kappa;P.gamma;P.tau;P.alpha;P.E0];

end

