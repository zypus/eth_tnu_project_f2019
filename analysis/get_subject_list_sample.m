function subjects = get_subject_list_sample(control,schizophrenia)
%GET_SUBJECT_LIST_SAMPLE Summary of this function goes here
%   Detailed explanation goes here
    file = fopen("../data/relevant_subjects.txt", "r");
    all_subjects = fscanf(file, "%9c");
    fclose(file);
    all_subjects = split(all_subjects);
    all_subjects = string(vertcat(all_subjects{:}));
    
    idx = 1;
    for i=1:size(all_subjects,1)
        sub = all_subjects(i);
        if startsWith(sub, "sub-1") && control > 0
            subjects(idx) = sub;
            control = control - 1;
            idx = idx + 1;
        elseif startsWith(sub, "sub-5") && schizophrenia > 0
            subjects(idx) = sub;
            schizophrenia = schizophrenia - 1;
            idx = idx + 1;
        end
    end
    
    subjects = subjects';
    
end



