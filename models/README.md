This directory contains code for running the pretrained Enformer and Basenji models on Basenji's data, as well as code for retraining Basenji models from scratch.  These models correspond to the following models referred to in our paper:
 * Enformer released
 * Basenji released
 * Basenji retrained
 * Basenji retrained with augmentation

See the README.md files within the Basenji and Enformer directories for details on how to run model retraining and model inference.

Note: Basenji's data are not included here due to size and cost.  Information on downloading their data is available here: https://github.com/calico/basenji/tree/master/manuscripts/cross2020. That link points to https://console.cloud.google.com/storage/browser/basenji_barnyard/data for downloading the data.  For this analysis, only the human data is required.

Additionally, `predict_test_set.py` script outputs inference results as large ground truth (`model_best_test_set_gt.pickle`) and predicted (`model_best_test_set_pred.pickle`) files, which are used in the analysis done by `Main.ipynb` in the top level directory.  However, these files are not included here due to size (~74GB each, two files per model for which inference was run).  Please contact the authors if you would like to have access these files.