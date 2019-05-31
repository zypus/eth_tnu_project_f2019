function P = load_model(name)
    model = load("../data/matlab/models/" + name + ".mat");
    P = model.P;
end

