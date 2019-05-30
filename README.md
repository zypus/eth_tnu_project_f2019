# ETH TNU Translational Neuromodeling Course Project F2019
Dissociating schizophrenia patients from controls (using stopsignal fMRI data OR simulated BOLD signals). Semester project for the Translational Neuromodeling ETH Course F2019

## Structure

1. Original data is obtained via some shell scripts (see [Original Data](#Original-Data) for details and caveats)
2. Timeseries extraction is performed via the python code in [preprocessing/extract_roi_timeseries.py](https://github.com/zypus/eth_tnu_project_f2019/tree/master/preprocessing/extract_roi_timeseries.py)
(3. The selection of ROI were informed by some analysis in R, which depends on preliminary output from the python code. Hence I have run both scripts back and forth a bit.)
4. DCMs were defined with [analysis/specify_models.m](https://github.com/zypus/eth_tnu_project_f2019/tree/master/analysis/specify_models.m)
5. Data simulation, DCM fitting and result are then generated with [analysis/main.m](https://github.com/zypus/eth_tnu_project_f2019/tree/master/analysis/main.m)

## BOLD timeseries extraction

This part is most likely the most error prone step of the setup, because I am yet lacking the required expertise to do this correctly.

## Original Data

None of the orignal data is required to iteract with the Matlab code all the derived data can be found in [data/](https://github.com/zypus/eth_tnu_project_f2019/tree/master/data).

However if you desire aquire the original data, be warned that the download scipts are written for a unix-shell only at the moment and require the [aws cmdline utility](https://aws.amazon.com/cli/).
Though an Batch equivalent for Windows should not be too difficult to come up with.

So in order to download the orignal dataset [ds000030](https://openneuro.org/datasets/ds000030/versions/00016) from openneuro. Execute the following code on the commandline:

```bash
cd data.nosync
# downloads only the relevant data from [legacy.openfmri.org](https://legacy.openfmri.org/s3-browser/?prefix=ds000030/ds000030_R1.0.5/uncompressed/) around 21GB
./data_aquisition.sh
# replaces "n/a" values with "0" in order to satisfy nilearn
./fix_confounds.sh
```
