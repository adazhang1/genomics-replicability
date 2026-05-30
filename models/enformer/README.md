The top level github directory is available at https://github.com/adazhang1/genomics-replicability.

This directory contains code for running the pretrained Enformer model on Basenji's data.  Code was built by referencing released code available at https://github.com/google-deepmind/deepmind-research/tree/master/enformer

# Environment loading
`requirements.txt` contains the package requirements specified by Enformer.  Note that the versions specified here differ slightly from what was released by Enformer.  We experienced conflicts when using Conda to install the requirements, so we adjusted the versions slightly to resolve the issue.  You can use `conda create -n enformer -c bioconda -c conda-forge --file requirements.txt` to build a conda environment from this file.  Then make sure to `conda activate enformer` before running model inference

# Model inference
Use `predict_test_set.py` to run Enformer on the test data.  You must specify the directory that contains the test data.  For example,
`python predict_test_set.py <BASENJI_DATA_DIR>`

Note: Basenji's data are not included here due to size and cost.  Information on downloading their data is available here: https://github.com/calico/basenji/tree/master/manuscripts/cross2020. That link points to https://console.cloud.google.com/storage/browser/basenji_barnyard/data for downloading the data.  For this analysis, only the human data is required.

The `genome.fa` FASTA file, used in inference, is too large to host on github (3 GB).  This file is included in the more complete `./models/enformer` directory available at Hugging Face: https://huggingface.co/adazhang1/genomics-replicability/tree/main/models/enformer.

`predict_test_set.py` will save ground truth (`model_best_test_set_gt.pickle`) and predicted (`model_best_test_set_pred.pickle`) values to large .pickle files.  These files are used in the analysis done by `Main.ipynb` in the top level directory.  However, due to size (~69GB each), these files are not included here.  Please contact the authors if you would like to have access these files.