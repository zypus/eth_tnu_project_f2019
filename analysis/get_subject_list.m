function subjects = get_subject_list()
%GET_SUBJECT_LIST Summary of this function goes here
%   Detailed explanation goes here
    file = fopen("../data/relevant_subjects.txt", "r");
    subjects = fscanf(file, "%9c");
    fclose(file);
    subjects = split(subjects);
    subjects = string(vertcat(subjects{:}));
end

