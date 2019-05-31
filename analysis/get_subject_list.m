function subjects = get_subject_list()
%GET_SUBJECT_LIST Get a list of all subject ids
    file = fopen("../data/relevant_subjects.txt", "r");
    subjects = fscanf(file, "%9c");
    fclose(file);
    subjects = split(subjects);
    subjects = string(vertcat(subjects{:}));
end

