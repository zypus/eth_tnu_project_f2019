function [Pm] = merge_parameters(P, mask, new_P)
%FLATTEN_PARAMETERS Summary of this function goes here
%   Detailed explanation goes here

 params = flatten_parameters(P);
 flat_mask = flatten_parameters(mask);
  
 index = 1;
 for i=1:size(flat_mask,1)
     if flat_mask(i) == 1
         params(i) = new_P(index);
         index = index + 1;
     end
 end
 
 Pm = structure_parameters(params, P);
 
end
