function P = convert_to_parameters(parname, parvalue, template)
%CONVERT_TO_PARAMETERS Summary of this function goes here
%   Detailed explanation goes here
    flat_template = flatten_parameters(template);
    
    flat_P = zeros(size(flat_template,1),1);
    for i=1:size(parname,2)
        idx = int32(str2double(string(parname{i})));
        flat_P(idx) = parvalue{i};
    end
    
    P = structure_parameters(flat_P, template);
end

