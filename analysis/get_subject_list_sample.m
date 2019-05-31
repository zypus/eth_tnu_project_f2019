function subjects = get_subject_list_sample(Nc,Ns)
%GET_SUBJECT_LIST_SAMPLE Gets subject ids and ensures there are at most
%Nc many control subjects and Ns many schizophrenia subjects 
    file = fopen("../data/relevant_subjects.txt", "r");
    all_subjects = fscanf(file, "%9c");
    fclose(file);
    all_subjects = split(all_subjects);
    all_subjects = string(vertcat(all_subjects{:}));
    
    idx = 1;
    for i=1:size(all_subjects,1)
        sub = all_subjects(i);
        if startsWith(sub, "sub-1") && Nc > 0
            subjects(idx) = sub;
            Nc = Nc - 1;
            idx = idx + 1;
        elseif startsWith(sub, "sub-5") && Ns > 0
            subjects(idx) = sub;
            Ns = Ns - 1;
            idx = idx + 1;
        end
    end
    
    subjects = subjects';
    
end



