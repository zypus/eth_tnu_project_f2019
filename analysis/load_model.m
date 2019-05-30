function P = load_model(name)
%LOAD_MODEL Summary of this function goes here
%   Detailed explanation goes here
    model = load("../data/matlab/models/" + name + ".mat");
    P = model.P;
end

