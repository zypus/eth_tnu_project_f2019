import numpy as np
from nilearn import datasets
from nilearn.image import iter_img
from nilearn.input_data import NiftiMapsMasker
from nilearn.connectome import ConnectivityMeasure

from nilearn import plotting, image
import glob
import os.path
import scipy.io as sio
import pandas as pd

from nilearn import decomposition
from nilearn.plotting import find_xyz_cut_coords

#aal_atlas = datasets.fetch_atlas_aal()
#len(aal_atlas["labels"])
#yeo_atlas = datasets.fetch_atlas_yeo_2011()
#yeo_atlas["thick_17"]
#len(yeo_atlas["labels"])
harvard_atlas = datasets.fetch_atlas_harvard_oxford('cort-prob-2mm')

msdl_atlas = datasets.fetch_atlas_msdl()

atlas_name = "harvard_oxford"

atlases = {"harvard_oxford": harvard_atlas, "msdl": msdl_atlas}
atlas = atlases[atlas_name]

masker = NiftiMapsMasker(maps_img=atlas["maps"], standardize=False, verbose=0, t_r=2, low_pass=0.1, high_pass=0.01, smoothing_fwhm=2)

subjects = []
for sub in glob.glob("data.nosync/ds000030/sub*"):
    sub_id = os.path.basename(sub)
    func_file = sub + f"/func/{sub_id}_task-stopsignal_bold_space-MNI152NLin2009cAsym_preproc.nii.gz"
    confounds_file = sub + f"/func/{sub_id}_task-stopsignal_bold_confounds-FIX.tsv"
    events_file = sub + f"/func/{sub_id}_task-stopsignal_events.tsv"
    if os.path.exists(func_file):
        subject = {"id": sub_id,
                   "func": func_file,
                   "confounds": confounds_file,
                   "events": events_file
                   }
        subjects.append(subject)
    else:
        print(f"Missing '{func_file}'")

# setup data file structure

os.makedirs(f"data/time_series/{atlas_name}", exist_ok=True)
os.makedirs("data/input", exist_ok=True)
os.makedirs("data/matlab", exist_ok=True)
os.makedirs("data/matlab/subjects", exist_ok=True)
os.makedirs("data/atlas", exist_ok=True)

# Extract time series

for sub in subjects:
    time_series_file = f"data/time_series/{atlas_name}/{sub['id']}_task_stopsignal_bold-{atlas_name.upper()}-time_series.txt"
    if os.path.exists(time_series_file):
        time_series = np.loadtxt(time_series_file)
    else:
        time_series = masker.fit_transform(sub["func"], confounds=sub["confounds"])
        np.savetxt(time_series_file, time_series)
    sub["time_series"] = time_series

region_name_file = f"data/time_series/{atlas_name}/region_names.txt"
np.savetxt(region_name_file, atlas["labels"][1:], fmt="%s")

# time_series = masker.fit_transform(subjects[0]["func"])
# print(np.shape(time_series))
# time_series = masker.fit_transform(subjects[3]["func"], confounds=subjects[3]["confounds"])

# prepare input data U

for sub in subjects:
    input_file = f"data/input/{sub['id']}_task_stopsignal_input.txt"
    if os.path.exists(input_file):
        u = np.loadtxt(input_file)
    else:
        events = pd.read_csv(sub["events"], sep="\t")
        filtered_events = events[events.trial_type.isin(["GO", "STOP"])]
        time_length = np.shape(sub["time_series"])[0]
        microtime = 20
        u1 = np.zeros(time_length * microtime)
        u2 = np.zeros(time_length * microtime)
        time_step = 2 / microtime
        impuls_length = 1.5
        for i in range(time_length * microtime):
            hit = (i * time_step <= filtered_events.onset + filtered_events.StopSignalDelay + impuls_length) & (i * time_step >= filtered_events.onset + filtered_events.StopSignalDelay)
            if any(hit):
                u1[i] = 1
                hits = filtered_events[hit]
                if any(hits.trial_type.isin(["STOP"]) & (i * time_step >= hits.StopSignalDelay) & (i * time_step <= filtered_events.onset + filtered_events.StopSignalDelay + 0.1)):
                    u2[i] = 1

        u = np.column_stack((u1, u2))

        np.savetxt(input_file, u)

    sub["input"] = u

# save data in matlab files

selected_regions = ["Inferior Temporal Gyrus, anterior division", "Insular Cortex", "Frontal Pole", "Inferior Frontal Gyrus, pars triangularis"]

for sub in subjects:
    selected_time_series = []
    for region in selected_regions:
        selected_time_series.append(sub["time_series"][:, atlas["labels"].index(region) - 1])
    confounds = pd.read_csv(sub["confounds"], sep="\t")
    mat_sub = {
        "id": sub["id"],
        "y": np.row_stack(selected_time_series),
        "u": np.transpose(sub["input"]),
        "x0": confounds.values
    }
    sio.savemat(f"data/matlab/subjects/{sub['id']}.mat", mat_sub)


# Compute brain atlas : DictLearning

comps = 20

dict_learning = decomposition.DictLearning(n_components=comps,
                                           verbose=1,
                                           random_state=0,
                                           n_epochs=1,
                                           mask_strategy='template')

dict_learning.fit([sub["func"] for sub in subjects], confounds=[sub["confounds"] for sub in subjects])

dict_img = dict_learning.components_img_

dict_img.to_filename('data/atlas/dict_learning.nii.gz')

plotting.plot_prob_atlas(dict_img, title='All Dict components')


# Experiments

atlas_img = image.clean_img(atlas["maps"], detrend=False, standardize=False, ensure_finite=True)
for i, cur_img in enumerate(image.iter_img(atlas_img)):
    label = atlas["labels"][i+1]
    plotting.plot_stat_map(cur_img, title=label, colorbar=False, output_file=f"figures/{label}.png")
plotting.show()

# region extraction

from nilearn.regions import RegionExtractor

extractor = RegionExtractor(dict_img, threshold=0.5,
                            thresholding_strategy='ratio_n_voxels',
                            extractor='local_regions',
                            standardize=False, min_region_size=1350)
# Just call fit() to process for regions extraction
extractor.fit()
# Extracted regions are stored in regions_img_
regions_extracted_img = extractor.regions_img_
# Each region index is stored in index_
regions_index = extractor.index_
# Total number of regions extracted
n_regions_extracted = regions_extracted_img.shape[-1]

# Visualization of region extraction results
title = ('%d regions are extracted from %d components.'
         '\nEach separate color of region indicates extracted region'
         % (n_regions_extracted, comps))
plotting.plot_prob_atlas(regions_extracted_img, view_type='filled_contours',
                         title=title)
plotting.show()


# First we need to do subjects timeseries signals extraction and then estimating
# correlation matrices on those signals.
# To extract timeseries signals, we call transform() from RegionExtractor object
# onto each subject functional data stored in func_filenames.
# To estimate correlation matrices we import connectome utilities from nilearn
from nilearn.connectome import ConnectivityMeasure

correlations = []
# Initializing ConnectivityMeasure object with kind='correlation'
connectome_measure = ConnectivityMeasure(kind='correlation')
for sub in subjects:
    # call transform from RegionExtractor object to extract timeseries signals
    timeseries_each_subject = extractor.transform(sub["func"], confounds=sub["confounds"])
    sub["time_series2"] = timeseries_each_subject
    # call fit_transform from ConnectivityMeasure object
    correlation = connectome_measure.fit_transform([timeseries_each_subject])
    # saving each subject correlation to correlations
    correlations.append(correlation)

# Mean of all correlations
import numpy as np
mean_correlations = np.mean(correlations, axis=0).reshape(n_regions_extracted,
                                                          n_regions_extracted)

title = 'Correlation between %d regions' % n_regions_extracted

# First plot the matrix
display = plotting.plot_matrix(mean_correlations, vmax=1, vmin=-1,
                               colorbar=True, title=title)

# Then find the center of the regions and plot a connectome
regions_img = regions_extracted_img
coords_connectome = plotting.find_probabilistic_atlas_cut_coords(regions_img)

plotting.plot_connectome(mean_correlations, coords_connectome,
                         edge_threshold='90%', title=title)
plotting.show()



# Find coresponding atlas regions

import operator

for sr in selected_regions:
    region_img = image.index_img(atlas_img, atlas["labels"].index(sr) - 1)
    region_coord = np.array(find_xyz_cut_coords(region_img))

    my_regions = {i: np.sum(np.square(np.array(find_xyz_cut_coords(img))-region_coord)) for i, img in enumerate(image.iter_img(regions_extracted_img))}

    sorted_x = sorted(my_regions.items(), key=operator.itemgetter(1))

    print(sr)
    print(sorted_x)

display = plotting.plot_stat_map(image.index_img(regions_extracted_img, 60))
display.add_overlay(image.index_img(atlas_img, atlas["labels"].index(selected_regions[3]) - 1))
plotting.show()


selected_regions2 = [28, 62, 51, 60]

for sub in subjects:
    selected_time_series = []
    for region in selected_regions2:
        selected_time_series.append(sub["time_series2"][:, region])
    confounds = pd.read_csv(sub["confounds"], sep="\t")
    mat_sub = {
        "id": sub["id"],
        "y": np.row_stack(selected_time_series),
        "u": np.transpose(sub["input"]),
        "x0": confounds.values
    }
    sio.savemat(f"data/matlab/subjects/{sub['id']}.mat", mat_sub)