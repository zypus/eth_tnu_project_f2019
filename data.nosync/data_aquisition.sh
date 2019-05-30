cat relevant_subjects.txt | xargs -n 1 -I % -P 8 aws s3 cp --no-sign-request s3://openneuro/ds000030/ds000030_R1.0.5/uncompressed/derivatives/fmriprep/%/func/%_task-stopsignal_bold_space-MNI152NLin2009cAsym_preproc.nii.gz ds000030/%/func/
cat relevant_subjects.txt | xargs -n 1 -I % -P 8 aws s3 cp --no-sign-request s3://openneuro/ds000030/ds000030_R1.0.5/uncompressed/derivatives/fmriprep/%/func/%_task-stopsignal_bold_confounds.tsv ds000030/%/func/
cat relevant_subjects.txt | xargs -n 1 -I % -P 8 aws s3 cp --no-sign-request s3://openneuro/ds000030/ds000030_R1.0.5/uncompressed/%/func/%_task-stopsignal_events.tsv ds000030/%/func/

