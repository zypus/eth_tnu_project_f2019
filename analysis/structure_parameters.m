function [P] = structure_parameters(pf, template)
%STRUCTURE_PARAMETERS Converts vector of parameters into parameter
%structure (inverse of FLATTEN_PARAMETERS)
%   pf - Vector of double values
%   template - Defines the sizes of the individual matrices in the
%   structure, this is usually just a reference parameter structure, as the
%   values inside don't matter just the dimensions are important


 A = size(template.A);
 B = size(template.B);
 C = size(template.C);
 D1 = size(template.D1);
 D2 = size(template.D2);

 offset = 1;
 P.A = reshape(pf(offset:A(1)*A(2)), A);
 offset=offset+A(1)*A(2);
 P.B = reshape(pf(offset:offset+B(1)*B(2)-1), B);
 offset=offset+B(1)*B(2);
 P.C = reshape(pf(offset:offset+C(1)*C(2)-1), C);
 offset=offset+C(1)*C(2);
 P.D1 = reshape(pf(offset:offset+D1(1)*D1(2)-1), D1);
 offset=offset+D1(1)*D1(2);
 P.D2 = reshape(pf(offset:offset+D2(1)*D2(2)-1), D2);
 offset=offset+D2(1)*D2(2);
 
%  P.A = P.A';
%  P.B = P.B';
%  P.C = P.C';
%  P.D = P.D';
 
 P.kappa = pf(offset);
 P.gamma = pf(offset+1);
 P.tau = pf(offset+2);
 P.alpha = pf(offset+3);
 P.E0 = pf(offset+4);
end
