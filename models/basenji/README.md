The top level github directory is available at https://github.com/adazhang1/genomics-replicability.

This directory contains the released, pretrained Basenji model as well as code for retraining Basenji from scratch.  Basenji code was downloaded on Jan 31, 2025

# Environment loading
`prespecified.yml` is the prespecified environment released by Basenji.  Their instructions for installation are available at https://github.com/calico/basenji/tree/master.  You can use `conda env create -f environment.yml` to build a conda environment from this file

# Model retraining
Use `basenji_train.py` to train Basenji models from scratch.  You must specify the parameters file and the directory containing the data.  For example,
`python basenji_train.py ./original_all/params.json <BASENJI_DATA_DIR>`

Note: Basenji's data are not included here due to size and cost.  Information on downloading their data is available here: https://github.com/calico/basenji/tree/master/manuscripts/cross2020. That link points to https://console.cloud.google.com/storage/browser/basenji_barnyard/data for downloading the data.  For this analysis, only the human data is required.

`./original_all/params.json`

# Model inference
Use `predict_test_set.py` to run a trained model on test data.  You must specify the model directory, which should contain a `params.json` that defines the model architecture and a .h5 model file that contains the trained model weights.  You must also point to the directory containing the test data.  For example,
`python predict_test_set.py ./released_model/ <BASENJI_DATA_DIR>`

As stated above, Basenji's data are not included here due to size and cost.  See above on how to download Basenji's data.

The trained models and model check points, stored as .h5 files, are too large to host on github.  A more complete `./models/basenji` directory is available at Hugging Face: https://huggingface.co/adazhang1/genomics-replicability/tree/main/models/basenji.

Note: `predict_test_set.py` will save ground truth (`model_best_test_set_gt.pickle`) and predicted (`model_best_test_set_pred.pickle`) values to large .pickle files.  These files are used in the analysis done by `Main.ipynb` in the top level directory.  However, due to size (~69GB each), these files are not included here.  Please contact the authors if you would like to have access these files.

We have included three models in our analysis:
 * `./released_model` contains the pretrained model released by Basenji.
 * `./original` contains a model that was retrained using the code and data released by Basenji.  However, upon further inspection, the `params.json` that was released does not include augmentation, which was described as part of the training procedure.
 * `./augmented` contains a model that was retrained after editing the original `params.json` file to include reverse compliment and +/-3bp shift augmentation.