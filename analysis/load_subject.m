function sub = load_subject(id)
%LOAD_SUBJECT Summary of this function goes here
%   Detailed explanation goes here
    sub = load("../data/matlab/subjects/" + id + ".mat");
    sub.y = double(sub.y);
end

