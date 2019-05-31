function sub = load_subject(id)
    sub = load("../data/matlab/subjects/" + id + ".mat");
    sub.y = double(sub.y);
end

