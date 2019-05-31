function save_model(name, P)
    if (exist("../data/matlab/models", "dir") ~= 7)
        mkdir("../data/matlab", "models");
    end
    save(strcat("../data/matlab/models/", name, ".mat"), "P");
end

